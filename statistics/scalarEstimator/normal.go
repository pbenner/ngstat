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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NormalEstimator struct {
  *scalarDistribution.NormalDistribution
  StdEstimator
  SigmaMin float64
}

func NewNormalEstimator(mu, sigma Scalar, sigmaMin float64) (*NormalEstimator, error) {
  if dist, err := scalarDistribution.NewNormalDistribution(mu, sigma); err != nil {
    return nil, err
  } else {
    r := NormalEstimator{}
    r.NormalDistribution = dist
    return &r, nil
  }
}

func (obj *NormalEstimator) CloneScalarEstimator() ScalarEstimator {
  r := NormalEstimator{}
  r.NormalDistribution = obj.NormalDistribution.Clone()
  r.SigmaMin = obj.SigmaMin
  r.x        = obj.x
  return &r
}

func (obj *NormalEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  if p.IsNil() {
    p = NewThreadPool(1, 1)
  }
  g := p.NewJobGroup()
  x := obj.x

  // allocate memory
  //////////////////////////////////////////////////////////////////////////////
  sum_g_ := make([]float64, p.NumberOfThreads())
  sum_m_ := make([]float64, p.NumberOfThreads())
  sum_s_ := make([]float64, p.NumberOfThreads())
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_g_[i] = 0.0
    sum_m_[i] = 0.0
    sum_s_[i] = 0.0
  }
  sum_g := 0.0
  sum_m := 0.0
  sum_s := 0.0
  // rescale gamma
  //////////////////////////////////////////////////////////////////////////////
  gamma_max := math.Inf(-1)
  for i := 0; i < gamma.Dim(); i++ {
    if g := gamma.At(i).GetValue(); gamma_max < g {
      gamma_max = g
    }
  }
  // compute mu
  //////////////////////////////////////////////////////////////////////////////
  p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
    id := p.GetThreadId()
    g  := math.Exp(gamma.At(i).GetValue() - gamma_max)
    y  := x[i].GetValue()
    // sum over gamma
    sum_g_[id] += g
    // sum over gamma*x
    sum_m_[id] += g*y
    return nil
  })
  if err := p.Wait(g); err != nil {
    return err
  }
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_g += sum_g_[i]
    sum_m += sum_m_[i]
  }
  // new mu value
  m := sum_m/sum_g
  // compute sigma
  //////////////////////////////////////////////////////////////////////////////
  p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
    id := p.GetThreadId()
    g  := math.Exp(gamma.At(i  ).GetValue() - gamma_max)
    y  := x[i].GetValue()
    // sum over (x-mu)^2
    sum_s_[id] += g*(y-m)*(y-m)
    return nil
  })
  if err := p.Wait(g); err != nil {
    return err
  }
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_s += sum_s_[i]
  }
  s := math.Sqrt(sum_s/sum_g)

  if s < obj.SigmaMin {
    s = obj.SigmaMin
  }
  // new parameters
  mu    := NewScalar(obj.ScalarType(), m)
  sigma := NewScalar(obj.ScalarType(), s)

  if t, err := scalarDistribution.NewNormalDistribution(mu, sigma); err != nil {
    return err
  } else {
    *obj.NormalDistribution = *t
  }
  return nil
}


func (estimator *NormalEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := estimator.SetData(x, len(x)); err != nil {
    return err
  }
  return estimator.Estimate(gamma, p)
}

func (estimator *NormalEstimator) GetEstimate() ScalarDistribution {
  return estimator.NormalDistribution
}
