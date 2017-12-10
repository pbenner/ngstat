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

package generic

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "bytes"
import   "math"

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Mixture struct {
  LogWeights Vector
  // state
  t1 Scalar
  t2 Scalar
}

/* -------------------------------------------------------------------------- */

func NewMixture(weights Vector) (*Mixture, error) {
  for i := 0; i < weights.Dim(); i++ {
    if weights.At(i).GetValue() < 0.0 {
      return nil, fmt.Errorf("weights must be positive")
    }
  }
  r := Mixture{}
  r.LogWeights = weights.CloneVector()
  r.LogWeights.Map(func(x Scalar) { x.Log(x) })
  r.t1 = NullScalar(weights.ElementType())
  r.t2 = NullScalar(weights.ElementType())
  r.normalize()
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) Clone() *Mixture {
  r := Mixture{}
  r.LogWeights = obj.LogWeights.CloneVector()
  r.t1         = obj.t1.CloneScalar()
  r.t2         = obj.t2.CloneScalar()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) normalize() {
  t1 := obj.t1
  t2 := obj.t2
  t1.SetValue(math.Inf(-1))
  for i := 0; i < obj.LogWeights.Dim(); i++ {
    t1.LogAdd(t1, obj.LogWeights.At(i), t2)
  }
  for i := 0; i < obj.LogWeights.Dim(); i++ {
    obj.LogWeights.At(i).Sub(obj.LogWeights.At(i), t1)
  }
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) NComponents() int {
  return obj.LogWeights.Dim()
}

func (obj *Mixture) ScalarType() ScalarType {
  return obj.LogWeights.ElementType()
}

func (obj *Mixture) LogPdf(r Scalar, data MixtureDataRecord) error {
  r.SetValue(0.0)
  t1 := obj.t1
  t2 := obj.t2
  // likelihood at position i
  r.SetValue(math.Inf(-1))
  // loop over components
  for j := 0; j < obj.NComponents(); j++ {
    if err := data.LogPdf(t1, j); err != nil {
      return err
    }
    t1.Add(t1, obj.LogWeights.At(j))
    r.LogAdd(r, t1, t2)
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) GetParameters() Vector {
  return obj.LogWeights
}

func (obj *Mixture) SetParameters(parameters Vector) error {
  if parameters.Dim() != obj.LogWeights.Dim() {
    return fmt.Errorf("invalid set of parameters")
  }
  for i := 0; i < obj.LogWeights.Dim(); i++ {
    obj.LogWeights.At(i).Set(parameters.At(i))
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) String() string {
  var buffer bytes.Buffer

  weights := obj.LogWeights.CloneVector()
  weights.Map(func(x Scalar) { x.Exp(x) })

  fmt.Fprintf(&buffer, "Mixture weights:\n%s\n", weights)

  return buffer.String()
}

/* -------------------------------------------------------------------------- */

func (obj *Mixture) ImportConfig(config ConfigDistribution, t ScalarType) error {
  weights, ok := config.GetParametersAsVector(obj.ScalarType()); if ! ok {
    return fmt.Errorf("invalid config file")
  }
  if tmp, err := NewMixture(weights); err != nil {
    return err
  } else {
    *obj = *tmp
  }
  return nil
}

func (obj *Mixture) ExportConfig() ConfigDistribution {

  weights := obj.LogWeights.CloneVector()
  weights.Map(func(x Scalar) { x.Exp(x) })

  return NewConfigDistribution("generic mixture", weights)
}
