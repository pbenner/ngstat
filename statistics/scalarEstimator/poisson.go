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

//import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type PoissonEstimator struct {
  *scalarDistribution.PoissonDistribution
  StdEstimator
}

func NewPoissonEstimator(lambda Scalar) (*PoissonEstimator, error) {
  if dist, err := scalarDistribution.NewPoissonDistribution(lambda); err != nil {
    return nil, err
  } else {
    r := PoissonEstimator{}
    r.PoissonDistribution = dist
    return &r, nil
  }
}

func (obj *PoissonEstimator) CloneScalarEstimator() ScalarEstimator {
  r := PoissonEstimator{}
  r.PoissonDistribution = obj.PoissonDistribution.Clone()
  r.x = obj.x
  return &r
}

func (obj *PoissonEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  if p.IsNil() {
    p = NewThreadPool(1, 1)
  }
  g := p.NewJobGroup()
  x := obj.x

  // allocate memory
  //////////////////////////////////////////////////////////////////////////////
  sum_mu_ := make([]float64, p.NumberOfThreads())
  sum_g_  := make([]float64, p.NumberOfThreads())
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_mu_[i] = math.Inf(-1)
    sum_g_ [i] = math.Inf(-1)
  }
  sum_mu := math.Inf(-1)
  sum_g  := math.Inf(-1)

  // loop over observations
  //////////////////////////////////////////////////////////////////////////////
  p.AddRangeJob(0, gamma.Dim(), g, func(k int, p ThreadPool, erf func() error) error {
    if !math.IsInf(gamma.At(k).GetValue(), -1) {
      id := p.GetThreadId()
      sum_mu_[id] = LogAdd(sum_mu_[id], gamma.At(k).GetValue() + math.Log(x[k].GetValue()))
      sum_g_ [id] = LogAdd(sum_g_ [id], gamma.At(k).GetValue())
    }
    return nil
  })
  if err := p.Wait(g); err != nil {
    return err
  }
  // sum up partial results
  //////////////////////////////////////////////////////////////////////////////
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_mu = LogAdd(sum_mu, sum_mu_[i])
    sum_g  = LogAdd(sum_g,  sum_g_ [i])
  }
  // compute new means
  //////////////////////////////////////////////////////////////////////////////
  mu := NewBareReal(math.Exp(sum_mu - sum_g))

  //////////////////////////////////////////////////////////////////////////////
  if t, err := scalarDistribution.NewPoissonDistribution(mu); err != nil {
    return err
  } else {
    *obj.PoissonDistribution = *t
  }
  return nil
}


func (estimator *PoissonEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := estimator.SetData(x, len(x)); err != nil {
    return err
  }
  return estimator.Estimate(gamma, p)
}

func (estimator *PoissonEstimator) GetEstimate() ScalarDistribution {
  return estimator.PoissonDistribution
}
