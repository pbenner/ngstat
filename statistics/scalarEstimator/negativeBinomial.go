/* Copyright (C) 2017 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package scalarEstimator

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NegativeBinomialEstimator struct {
  *scalarDistribution.NegativeBinomialDistribution
  StdEstimator
}

func NewNegativeBinomialEstimator(r, p, pseudocount Scalar) (*NegativeBinomialEstimator, error) {
  if dist, err := scalarDistribution.NewNegativeBinomialDistribution(r, p, pseudocount); err != nil {
    return nil, err
  } else {
    r := NegativeBinomialEstimator{}
    r.NegativeBinomialDistribution = dist
    return &r, nil
  }
}

func (obj *NegativeBinomialEstimator) CloneScalarEstimator() ScalarEstimator {
  r := NegativeBinomialEstimator{}
  r.NegativeBinomialDistribution = obj.NegativeBinomialDistribution.Clone()
  r.x = obj.x
  return &r
}

func (obj *NegativeBinomialEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  if p.IsNil() {
    p = NewThreadPool(1, 1)
  }
  g := p.NewJobGroup()
  x := obj.x

  sum_k_ := make([]float64, p.NumberOfThreads())
  sum_r_ := make([]float64, p.NumberOfThreads())
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_k_[i] = math.Inf(-1)
    sum_r_[i] = math.Inf(-1)
  }
  // loop over observations
  if gamma == nil {
    p.AddRangeJob(0, len(x), g, func(k int, p ThreadPool, erf func() error) error {
      if !math.IsInf(gamma.At(k).GetValue(), -1) {
        id := p.GetThreadId()
        sum_k_[id] = LogAdd(sum_k_[id], gamma.At(k).GetValue() + math.Log(x[k].GetValue() + obj.Pseudocount.GetValue()))
        sum_r_[id] = LogAdd(sum_r_[id], gamma.At(k).GetValue() + math.Log(obj.R.GetValue()))
      }
      return nil
    })
  } else {
    p.AddRangeJob(0, len(x), g, func(k int, p ThreadPool, erf func() error) error {
      id := p.GetThreadId()
      sum_k_[id] = LogAdd(sum_k_[id], math.Log(x[k].GetValue() + obj.Pseudocount.GetValue()))
      sum_r_[id] = LogAdd(sum_r_[id], math.Log(obj.R.GetValue()))
      return nil
    })
  }
  if err := p.Wait(g); err != nil {
    return err
  }
  // sum up partial results
  sum_k := math.Inf(-1)
  sum_r := math.Inf(-1)
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_k = LogAdd(sum_k, sum_k_[i])
    sum_r = LogAdd(sum_r, sum_r_[i])
  }
  if math.IsInf(sum_r, -1) && math.IsInf(sum_k, -1) {
    return fmt.Errorf("negative binomial parameter estimation failed")
  }
  t := obj.ScalarType()
  q := NewScalar(t, 0.0)
  q.SetValue(math.Exp(sum_k - LogAdd(sum_r, sum_k)))
  if t, err := scalarDistribution.NewNegativeBinomialDistribution(obj.R, q, obj.Pseudocount); err != nil {
    return err
  } else {
    *obj.NegativeBinomialDistribution = *t
  }
  return nil
}

func (estimator *NegativeBinomialEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := estimator.SetData(x, len(x)); err != nil {
    return err
  }
  return estimator.Estimate(gamma, p)
}

func (estimator *NegativeBinomialEstimator) GetEstimate() ScalarDistribution {
  return estimator.NegativeBinomialDistribution
}
