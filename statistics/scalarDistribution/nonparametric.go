/* Copyright (C) 2016 Philipp Benner
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

package scalarDistribution

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import   "sort"

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type NonparametricDistribution struct {
  MargDensity   Vector  // p(x_i)
  Delta       []float64
  X           []float64
  Xmap          map[float64]int
}

/* -------------------------------------------------------------------------- */

func NewNonparametricDistribution(x, y []float64) (*NonparametricDistribution, error) {
  if len(x) != len(y) {
    return nil, fmt.Errorf("dimensions do not match")
  }
  if r, err := NullNonparametricDistribution(x); err != nil {
    return nil, err
  } else {
    for i := 0; i < len(y); i++ {
      r.MargDensity.At(i).SetValue(y[i])
    }
    return r, nil
  }
}

func NullNonparametricDistribution(x []float64) (*NonparametricDistribution, error) {
  r := &NonparametricDistribution{}
  r.MargDensity = NullVector(BareRealType, len(x))
  r.Delta = make([]float64, len(x))
  r.X     = make([]float64, len(x))
  // map values to indices
  r.Xmap  = make(map[float64]int)
  for i, v := range x {
    r.X[i]    = v
    r.Xmap[v] = i
  }
  for i := 0; i < len(r.X)-1; i++ {
    r.Delta[i] = r.X[i+1] - r.X[i]
  }
  if n := len(r.Delta); n > 1 {
    // set last delta equal to the second last
    r.Delta[n-1] = r.Delta[n-2]
  } else {
    // only one value present, choose an arbitrary delta of one
    r.Delta[0] = 1
  }
  return r, nil
}

/* -------------------------------------------------------------------------- */

func (dist *NonparametricDistribution) Clone() *NonparametricDistribution {
  delta := make([]float64, len(dist.Delta)); copy(delta, dist.Delta)
  x     := make([]float64, len(dist.X))    ; copy(x,     dist.X)
  xmap  := make(map[float64]int)
  for k, v := range dist.Xmap {
    xmap[k] = v
  }
  return &NonparametricDistribution{
    MargDensity: dist.MargDensity.CloneVector(),
    Delta      : delta,
    X          : x,
    Xmap       : xmap }
}

func (dist *NonparametricDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *NonparametricDistribution) ScalarType() ScalarType {
  return dist.MargDensity.ElementType()
}

func (dist *NonparametricDistribution) Index(x_ Scalar) (int, error) {
  x := x_.GetValue()

  if idx, ok := dist.Xmap[x]; ok {
    return idx, nil
  } else {
    if x < dist.X[0] && x > dist.X[len(dist.X)-1]+dist.Delta[len(dist.Delta)-1] {
      return -1, fmt.Errorf("value `%v' is out of range", x)
    } else {
      idx = sort.SearchFloat64s(dist.X, x)
      return idx-1, nil
    }
  }
}

func (dist *NonparametricDistribution) LogPdf(r Scalar, y Scalar) error {
  r.SetValue(0.0)
  // add p(x_0)
  if i1, err := dist.Index(y); err != nil {
    r.SetValue(math.Inf(-1))
  } else {
    r.Add(r, dist.MargDensity.At(i1))
  }
  return nil
}

func (dist *NonparametricDistribution) Pdf(r Scalar, y Scalar) error {
  if err := dist.LogPdf(r, y); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

func (dist *NonparametricDistribution) LogEntropy() Scalar {
  sum := NewScalar(BareRealType, math.Inf(-1))
  t1  := NewBareReal(0.0)
  t2  := NewBareReal(0.0)
  for i := 0; i < dist.MargDensity.Dim(); i++ {
    t1.SetValue(math.Log(dist.Delta[i]))
    // t1 = log f(x)dx
    t1.Add(t1, dist.MargDensity.At(i))
    // t2 = log f(x)dx
    t2.Set(t1)
    // t1 = - log f(x)dx
    t1.Neg(t1)
    // t1 = log(-log f(x)dx)
    t1.Log(t1)
    // t1 = log(-log f(x)dx) + log f(x)dx = log[-f(x)dx log f(x)dx]
    t1.Add(t1, t2)
    sum.LogAdd(sum, t1, t2)
  }
  return sum
}

/* -------------------------------------------------------------------------- */

func (dist NonparametricDistribution) GetParameters() Vector {
  return dist.MargDensity
}

func (dist *NonparametricDistribution) SetParameters(parameters Vector) error {
  dist.MargDensity.Set(parameters)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *NonparametricDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    n := len(parameters)/2
    x := make([]float64, n)
    y := make([]float64, n)

    for i := 0; i < n; i++ {
      x[i] = parameters[0+i]
      y[i] = parameters[n+i]
    }

    if tmp, err := NewNonparametricDistribution(x, y); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *NonparametricDistribution) ExportConfig() ConfigDistribution {

  parameters := NewVector(BareRealType, dist.X)
  parameters  = parameters.AppendVector(dist.MargDensity)

  return NewConfigDistribution("nonparametric distribution", parameters)
}
