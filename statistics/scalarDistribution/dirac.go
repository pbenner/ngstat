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

//import   "fmt"
import   "io"
import   "math"

import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/statistics/config"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type DiracDistribution struct {
  X Scalar
}

/* -------------------------------------------------------------------------- */

func NewDiracDistribution(x Scalar) (*DiracDistribution, error) {
  return &DiracDistribution{x.CloneScalar()}, nil
}

/* -------------------------------------------------------------------------- */

func (dist *DiracDistribution) Clone() *DiracDistribution {
  return &DiracDistribution{dist.X.CloneScalar()}
}

func (dist *DiracDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *DiracDistribution) ScalarType() ScalarType {
  return dist.X.Type()
}

func (dist *DiracDistribution) LogPdf(r Scalar, x Scalar) error {
  if x.GetValue() == dist.X.GetValue() {
    r.SetValue(0.0)
  } else {
    r.SetValue(math.Inf(-1))
  }
  return nil
}

func (dist *DiracDistribution) Pdf(r Scalar, x Scalar) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *DiracDistribution) GetParameters() Vector {
  p := NullVector(dist.ScalarType(), 1)
  p.At(0).Set(dist.X)
  return p
}

func (dist *DiracDistribution) SetParameters(parameters Vector) error {
  if tmp, err := NewDiracDistribution(parameters.At(0)); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *DiracDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "dirac distribution"); err != nil {
    return err
  }

  x := NewScalar(BareRealType, config.Parameters[0])

  if tmp, err := NewDiracDistribution(x); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *DiracDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("dirac distribution", dist.GetParameters())

  return config.Export(writer)
}
