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

import . "github.com/pbenner/ngstat/statistics/config"

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

func (dist *GeneralizedGammaDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "generalized gamma distribution"); err != nil {
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

  a := NewScalar(t, config.Parameters[0])
  d := NewScalar(t, config.Parameters[1])
  p := NewScalar(t, config.Parameters[2])
  q := NewScalar(t, config.Parameters[3])

  if tmp, err := NewGeneralizedGammaDistribution(a, d, p, q); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *GeneralizedGammaDistribution) Export(writer io.Writer) error {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  config := NewConfigDistribution("generalized gamma distribution", parameters)

  return config.Export(writer)
}
