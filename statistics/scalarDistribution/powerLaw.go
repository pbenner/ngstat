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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "io"

import . "github.com/pbenner/ngstat/statistics/config"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/distribution"

/* -------------------------------------------------------------------------- */

type PowerLawDistribution struct {
  distribution.PowerLawDistribution
}

/* -------------------------------------------------------------------------- */

func NewPowerLawDistribution(alpha, xmin, epsilon Scalar) (*PowerLawDistribution, error) {
  if dist, err := distribution.NewPowerLawDistribution(alpha, xmin, epsilon); err != nil {
    return nil, err
  } else {
    return &PowerLawDistribution{*dist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *PowerLawDistribution) Clone() *PowerLawDistribution {
  return &PowerLawDistribution{*dist.PowerLawDistribution.Clone()}
}

func (dist *PowerLawDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *PowerLawDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "power law distribution"); err != nil {
    return err
  }
  // determine scalar type
  t := BareRealType
  for _, arg := range args {
    switch v := arg.(type) {
    case ScalarType:
      t = v
    }
  }

  alpha   := NewScalar(t, config.Parameters[0])
  xmin    := NewScalar(t, config.Parameters[1])
  epsilon := NewScalar(t, config.Parameters[2])

  if tmp, err := NewPowerLawDistribution(alpha, xmin, epsilon); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *PowerLawDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("power law distribution", dist.GetParameters())

  return config.Export(writer)
}
