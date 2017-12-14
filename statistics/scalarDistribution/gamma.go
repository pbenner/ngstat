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

type GammaDistribution struct {
  distribution.GammaDistribution
  Pseudocount Scalar
  // state
  x Scalar
}

/* -------------------------------------------------------------------------- */

func NewGammaDistribution(alpha, beta, pseudocount Scalar) (*GammaDistribution, error) {
  if dist, err := distribution.NewGammaDistribution(alpha, beta); err != nil {
    return nil, err
  } else {
    return &GammaDistribution{*dist, pseudocount, NullScalar(alpha.Type()) }, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) Clone() *GammaDistribution {
  return &GammaDistribution{*dist.GammaDistribution.Clone(), dist.Pseudocount.CloneScalar(), dist.x.CloneScalar()}
}

func (dist *GammaDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) LogPdf(r Scalar, x Scalar) error {
  dist.x.Add(x, dist.Pseudocount)
  return dist.GammaDistribution.LogPdf(r, dist.x)
}

/* -------------------------------------------------------------------------- */

func (dist *GammaDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    alpha       := NewScalar(t, parameters[0])
    beta        := NewScalar(t, parameters[1])
    pseudocount := NewScalar(t, parameters[2])

    if tmp, err := NewGammaDistribution(alpha, beta, pseudocount); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
  return nil
}

func (dist *GammaDistribution) ExportConfig() ConfigDistribution {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  return NewConfigDistribution("scalar:gamma distribution", parameters)
}
