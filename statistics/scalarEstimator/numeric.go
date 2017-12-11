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

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/bfgs"
import   "github.com/pbenner/autodiff/algorithm/newton"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NumericEstimator struct {
  ScalarDistribution
  StdEstimator
  Method          string
  Epsilon         float64
  MaxIterations   int
}

/* -------------------------------------------------------------------------- */

func NewNumericEstimator(f ScalarDistribution) (*NumericEstimator, error) {
  r := NumericEstimator{}
  r.ScalarDistribution = f.CloneScalarDistribution()
  r.Method             = "newton"
  r.Epsilon            = 1e-8
  r.MaxIterations      = 20
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) Clone() *NumericEstimator {
  r, _ := NewNumericEstimator(obj.ScalarDistribution)
  r.Method  = obj.Method
  r.Epsilon = obj.Epsilon
  r.x       = obj.x
  return r
}

func (obj *NumericEstimator) CloneScalarEstimator() ScalarEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) GetParameters() Vector {
  return obj.ScalarDistribution.GetParameters()
}

func (obj *NumericEstimator) SetParameters(parameters Vector) error {
  return obj.ScalarDistribution.SetParameters(parameters)
}

/* -------------------------------------------------------------------------- */

func (obj *NumericEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  nt := p.NumberOfThreads()
  x  := obj.x
  n  := obj.n
  m  := obj.x.Dim()
  // create a copy of the density function
  f := make([]ScalarDistribution, nt)
  for i := 0; i < len(f); i++ {
    f[i] = obj.ScalarDistribution.CloneScalarDistribution()
  }
  constraints_f := func(variables Vector) bool {
    if err := f[0].SetParameters(variables); err != nil {
      return false
    }
    return true
  }
  // define the objective function
  objective_f := func(variables Vector) (Scalar, error) {
    // temporary variable
    t := NullVector(RealType, nt)
    s := NullVector(RealType, nt)
    r := NullVector(RealType, nt)
    for i := 0; i < len(f); i++ {
      if err := f[i].SetParameters(variables); err != nil {
        return nil, err
      }
    }
    g := p.NewJobGroup()
    p.AddRangeJob(0, m, g, func(k int, p ThreadPool, erf func() error) error {
      f := f   [p.GetThreadId()]
      t := t.At(p.GetThreadId())
      s := s.At(p.GetThreadId())
      r := r.At(p.GetThreadId())
      // stop if there was an error in another thread
      if erf() != nil {
        return nil
      }
      if !math.IsInf(gamma.At(k).GetValue(), -1) {
        if err := f.LogPdf(t, x.At(k)); err != nil {
          return err
        }
        s.Exp(gamma.At(k))
        t.Mul(t, s)
        r.Add(r, t)
      }
      return nil
    })
    p.Wait(g)
    // sum up results from all threads
    for i := 1; i < r.Dim(); i++ {
      r.At(0).Add(r.At(0), r.At(i))
    }
    r.At(0).Neg(r.At(0))
    r.At(0).Div(r.At(0), NewReal(float64(n)))
    return r.At(0), nil
  }
  // get parameters of the density function and convert
  // the scalar type to real
  theta_0 := obj.ScalarDistribution.GetParameters()
  theta_0  = AsVector(RealType, theta_0)

  var theta_n Vector
  var err error

  switch obj.Method {
    // execute optimization algorithm
  case "newton":
    theta_n, err = newton.RunMin(objective_f, theta_0,
      newton.Epsilon            {obj.Epsilon},
      newton.MaxIterations      {obj.MaxIterations},
      newton.Constraints        {constraints_f},
      newton.HessianModification{"LDL"})
  case "bfgs":
    theta_n, err = bfgs.Run(objective_f, theta_0,
      bfgs.Epsilon      {obj.Epsilon},
      bfgs.MaxIterations{obj.MaxIterations},
      bfgs.Constraints  {constraints_f})
  }
  if err != nil && err.Error() != "line search failed" {
    return err
  } else {
    // set parameters of the density function, but keep the
    // initial scalar type
    v := obj.GetParameters()
    v.Set(theta_n)
    obj.SetParameters(v)
  }
  return nil
}

func (obj *NumericEstimator) EstimateOnData(x Vector, gamma DenseBareRealVector, p ThreadPool) error {
  if err := obj.SetData(x, x.Dim()); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *NumericEstimator) GetEstimate() ScalarDistribution {
  return obj.ScalarDistribution
}
