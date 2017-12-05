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

type ExponentialDistribution struct {
  distribution.ExponentialDistribution
}

/* -------------------------------------------------------------------------- */

func NewExponentialDistribution(lambda Scalar) (*ExponentialDistribution, error) {
  if dist, err := distribution.NewExponentialDistribution(lambda); err != nil {
    return nil, err
  } else {
    return &ExponentialDistribution{*dist}, err
  }
}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) Clone() *ExponentialDistribution {
  return &ExponentialDistribution{*dist.ExponentialDistribution.Clone()}
}

func (dist *ExponentialDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *ExponentialDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "exponential distribution"); err != nil {
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

  lambda := NewScalar(t, config.Parameters[0])

  if tmp, err := NewExponentialDistribution(lambda); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *ExponentialDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("exponential distribution", dist.GetParameters())

  return config.Export(writer)
}
