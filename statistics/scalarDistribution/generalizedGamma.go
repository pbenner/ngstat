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

type GeneralizedGammaDistribution struct {
  distribution.GeneralizedGammaDistribution
  Pseudocount Scalar
  x           Scalar
}

/* -------------------------------------------------------------------------- */

func NewGeneralizedGammaDistribution(a, d, p, pseudocount Scalar) (*GeneralizedGammaDistribution, error) {
  if dist, err := distribution.NewGeneralizedGammaDistribution(a, d, p); err != nil {
    return nil, err
  } else {
    return &GeneralizedGammaDistribution{*dist, pseudocount, NullScalar(a.Type())}, err
  }
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) Clone() *GeneralizedGammaDistribution {
  return &GeneralizedGammaDistribution{*dist.GeneralizedGammaDistribution.Clone(), dist.Pseudocount.CloneScalar(), dist.x.CloneScalar()}
}

func (dist *GeneralizedGammaDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) LogPdf(r Scalar, x Scalar) error {
  dist.x.Add(x, dist.Pseudocount)
  return dist.GeneralizedGammaDistribution.LogPdf(r, dist.x)
}

/* -------------------------------------------------------------------------- */

func (dist *GeneralizedGammaDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if parameters, ok := config.GetParametersAsFloats(); !ok {
    return fmt.Errorf("invalid config file")
  } else {
    a := NewScalar(t, parameters[0])
    d := NewScalar(t, parameters[1])
    p := NewScalar(t, parameters[2])
    q := NewScalar(t, parameters[3])

    if tmp, err := NewGeneralizedGammaDistribution(a, d, p, q); err != nil {
      return err
    } else {
      *dist = *tmp
    }
    return nil
  }
}

func (dist *GeneralizedGammaDistribution) ExportConfig() ConfigDistribution {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  return NewConfigDistribution("scalar:generalized gamma distribution", parameters)
}
