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

import   "fmt"

import . "github.com/pbenner/ngstat/statistics"

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

func (dist *ExponentialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    lambda := NewScalar(t, parameters[0])

    if tmp, err := NewExponentialDistribution(lambda); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *ExponentialDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("scalar:exponential distribution", dist.GetParameters())
}
