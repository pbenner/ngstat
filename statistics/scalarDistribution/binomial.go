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

type BinomialDistribution struct {
  distribution.BinomialDistribution
}

/* -------------------------------------------------------------------------- */

func NewBinomialDistribution(theta Scalar, n int) (*BinomialDistribution, error) {
  if dist, err := distribution.NewBinomialDistribution(theta, n); err != nil {
    return nil, err
  } else {
    return &BinomialDistribution{*dist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) Clone() *BinomialDistribution {
  return &BinomialDistribution{*dist.BinomialDistribution.Clone()}
}

func (dist *BinomialDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *BinomialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    theta := NewScalar(t, parameters[0])
    n     := int(parameters[1])

    if tmp, err := NewBinomialDistribution(theta, n); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *BinomialDistribution) ExportConfig() ConfigDistribution {

  return NewConfigDistribution("binomial distribution", dist.GetParameters())
}
