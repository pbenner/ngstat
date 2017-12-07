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

func (dist *PowerLawDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    alpha   := NewScalar(t, parameters[0])
    xmin    := NewScalar(t, parameters[1])
    epsilon := NewScalar(t, parameters[2])

    if tmp, err := NewPowerLawDistribution(alpha, xmin, epsilon); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *PowerLawDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("power law distribution", dist.GetParameters())
}
