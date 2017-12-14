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

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/distribution"

/* -------------------------------------------------------------------------- */

type GevDistribution struct {
  distribution.GevDistribution
}

/* -------------------------------------------------------------------------- */

func NewGevDistribution(mu, sigma, xi Scalar) (*GevDistribution, error) {
  if dist, err := distribution.NewGevDistribution(mu, sigma, xi); err != nil {
    return nil, err
  } else {
    return &GevDistribution{*dist}, err
  }
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) Clone() *GevDistribution {
  return &GevDistribution{*dist.GevDistribution.Clone()}
}

func (dist *GevDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GevDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    mu    := NewScalar(t, parameters[0])
    sigma := NewScalar(t, parameters[1])
    xi    := NewScalar(t, parameters[2])

    if tmp, err := NewGevDistribution(mu, sigma, xi); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *GevDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("scalar:gev distribution", dist.GetParameters())
}
