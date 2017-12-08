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

type LogNormalEstimator struct {
  *scalarDistribution.LogNormalDistribution
  StdEstimator
  SigmaMin float64
}

func NewLogNormalEstimator(mu, sigma, pseudocount Scalar, sigmaMin float64) (*LogNormalEstimator, error) {
  if dist, err := scalarDistribution.NewLogNormalDistribution(mu, sigma, pseudocount); err != nil {
    return nil, err
  } else {
    r := LogNormalEstimator{}
    r.LogNormalDistribution = dist
    return &r, nil
  }
}

func (obj *LogNormalEstimator) CloneScalarEstimator() ScalarEstimator {
  r := LogNormalEstimator{}
  r.LogNormalDistribution = obj.LogNormalDistribution.Clone()
  r.SigmaMin = obj.SigmaMin
  r.x        = obj.x
  return &r
}

func (obj *LogNormalEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
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
  if gamma != nil {
    for i := 0; i < gamma.Dim(); i++ {
      if g := gamma.At(i).GetValue(); gamma_max < g {
      gamma_max = g
      }
    }
  }
  // compute mu
  //////////////////////////////////////////////////////////////////////////////
  if gamma == nil {
    p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      id := p.GetThreadId()
      // sum over x
      sum_m_[id] += math.Log(x[i].GetValue() + obj.Pseudocount.GetValue())
      return nil
    })
  } else {
    p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      id := p.GetThreadId()
      g  := math.Exp(gamma.At(i  ).GetValue() - gamma_max)
      y  := math.Log(x[i].GetValue() + obj.Pseudocount.GetValue())
      // sum over gamma
      sum_g_[id] += g
      // sum over gamma*x
      sum_m_[id] += g*y
      return nil
    })
  }
  if err := p.Wait(g); err != nil {
    return err
  }
  for i := 0; i < p.NumberOfThreads(); i++ {
    sum_g += sum_g_[i]
    sum_m += sum_m_[i]
  }
  if gamma == nil {
    sum_g = float64(len(x))
  }
  // new mu value
  m := sum_m/sum_g
  // compute sigma
  //////////////////////////////////////////////////////////////////////////////
  if gamma == nil {
    p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      id := p.GetThreadId()
      y  := math.Log(x[i].GetValue() + obj.Pseudocount.GetValue())
      // sum over (x-mu)^2
      sum_s_[id] += (y-m)*(y-m)
      return nil
    })
  } else {
    p.AddRangeJob(0, gamma.Dim(), g, func(i int, p ThreadPool, erf func() error) error {
      id := p.GetThreadId()
      g  := math.Exp(gamma.At(i  ).GetValue() - gamma_max)
      y  := math.Log(x[i].GetValue() + obj.Pseudocount.GetValue())
      // sum over gamma*(x-mu)^2
      sum_s_[id] += g*(y-m)*(y-m)
      return nil
    })
  }
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

  if t, err := scalarDistribution.NewLogNormalDistribution(mu, sigma, obj.Pseudocount); err != nil {
    return err
  } else {
    *obj.LogNormalDistribution = *t
  }
  return nil
}

func (estimator *LogNormalEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := estimator.SetData(x, len(x)); err != nil {
    return err
  }
  return estimator.Estimate(gamma, p)
}

func (estimator *LogNormalEstimator) GetEstimate() ScalarDistribution {
  return estimator.LogNormalDistribution
}
