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

type ParetoDistribution struct {
  distribution.ParetoDistribution
}

/* -------------------------------------------------------------------------- */

func NewParetoDistribution(lambda, kappa, epsilon Scalar) (*ParetoDistribution, error) {
  if dist, err := distribution.NewParetoDistribution(lambda, kappa, epsilon); err != nil {
    return nil, err
  } else {
    return &ParetoDistribution{*dist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) Clone() *ParetoDistribution {
  return &ParetoDistribution{*dist.ParetoDistribution.Clone()}
}

func (dist *ParetoDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *ParetoDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    lambda  := NewScalar(t, parameters[0])
    kappa   := NewScalar(t, parameters[1])
    epsilon := NewScalar(t, parameters[2])

    if tmp, err := NewParetoDistribution(lambda, kappa, epsilon); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *ParetoDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("scalar:pareto distribution", dist.GetParameters())
}
