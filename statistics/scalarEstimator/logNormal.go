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
  // parameters
  SigmaMin float64
  // state
  sum_g []float64
  sum_m []float64
  sum_s []float64
  gamma_max float64
}

/* -------------------------------------------------------------------------- */

func NewLogNormalEstimator(mu, sigma, pseudocount Scalar, sigmaMin float64) (*LogNormalEstimator, error) {
  if dist, err := scalarDistribution.NewLogNormalDistribution(mu, sigma, pseudocount); err != nil {
    return nil, err
  } else {
    r := LogNormalEstimator{}
    r.LogNormalDistribution = dist
    return &r, nil
  }
}

/* -------------------------------------------------------------------------- */

func (obj *LogNormalEstimator) Clone() *LogNormalEstimator {
  r := LogNormalEstimator{}
  r.LogNormalDistribution = obj.LogNormalDistribution.Clone()
  r.SigmaMin = obj.SigmaMin
  r.x        = obj.x
  return &r
}

func (obj *LogNormalEstimator) CloneScalarEstimator() ScalarEstimator {
  return obj.Clone()
}

func (obj *LogNormalEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
  return obj.Clone()
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *LogNormalEstimator) Initialize(p ThreadPool) error {
  obj.sum_g = make([]float64, p.NumberOfThreads())
  obj.sum_m = make([]float64, p.NumberOfThreads())
  obj.sum_s = make([]float64, p.NumberOfThreads())
  for i := 0; i < p.NumberOfThreads(); i++ {
    obj.sum_g[i] = 0.0
    obj.sum_m[i] = 0.0
    obj.sum_s[i] = 0.0
  }
  obj.gamma_max = 0.0
  return nil
}

func (obj *LogNormalEstimator) NewObservation(x, gamma Scalar, p ThreadPool) error {
  id := p.GetThreadId()
  if gamma == nil {
    x := math.Log(x.GetValue() + obj.Pseudocount.GetValue())
    obj.sum_m[id] += x
    obj.sum_s[id] += x*x
    obj.sum_g[id] += 1.0
  } else {
    x := math.Log(x.GetValue() + obj.Pseudocount.GetValue())
    g := math.Exp(gamma.GetValue() - obj.gamma_max)
    obj.sum_m[id] += g*x
    obj.sum_s[id] += g*x*x
    obj.sum_g[id] += g
  }
  return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *LogNormalEstimator) updateEstimate() error {
  sum_g := 0.0
  sum_m := 0.0
  sum_s := 0.0
  for i := 0; i < len(obj.sum_m); i++ {
    sum_m += obj.sum_m[i]
    sum_s += obj.sum_s[i]
    sum_g += obj.sum_g[i]
  }
  s1 := sum_m/float64(sum_g)
  s2 := sum_s/float64(sum_g)

  mu    := NewScalar(obj.ScalarType(), s1)
  sigma := NewScalar(obj.ScalarType(), math.Sqrt(s2 - s1*s1))

  if sigma.GetValue() < obj.SigmaMin {
    sigma.SetValue(obj.SigmaMin)
  }

  if t, err := scalarDistribution.NewLogNormalDistribution(mu, sigma, obj.Pseudocount); err != nil {
    return err
  } else {
    *obj.LogNormalDistribution = *t
  }
  obj.sum_g = nil
  obj.sum_m = nil
  obj.sum_s = nil
  return nil
}

func (obj *LogNormalEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  if p.IsNil() {
    p = NewThreadPool(1, 1)
  }
  g := p.NewJobGroup()
  x := obj.x

  // initialize estimator
  obj.Initialize(p)

  // rescale gamma
  //////////////////////////////////////////////////////////////////////////////
  if gamma != nil {
    obj.gamma_max = math.Inf(-1)
    for i := 0; i < gamma.Dim(); i++ {
      if g := gamma.At(i).GetValue(); obj.gamma_max < g {
        obj.gamma_max = g
      }
    }
  }
  // compute sigma
  //////////////////////////////////////////////////////////////////////////////
  if gamma == nil {
    p.AddRangeJob(0, len(x), g, func(i int, p ThreadPool, erf func() error) error {
      obj.NewObservation(x[i], nil, p)
      return nil
    })
  } else {
    p.AddRangeJob(0, len(x), g, func(i int, p ThreadPool, erf func() error) error {
      obj.NewObservation(x[i], gamma.At(i), p)
      return nil
    })
  }
  if err := p.Wait(g); err != nil {
    return err
  }
  // update estimate
  if err := obj.updateEstimate(); err != nil {
    return err
  }
  return nil
}

func (obj *LogNormalEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := obj.SetData(x, len(x)); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *LogNormalEstimator) GetEstimate() ScalarDistribution {
  if obj.sum_m != nil {
    obj.updateEstimate()
  }
  return obj.LogNormalDistribution
}
