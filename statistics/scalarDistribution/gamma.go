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

func (dist *GammaDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "gamma distribution"); err != nil {
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

  alpha       := NewScalar(t, config.Parameters[0])
  beta        := NewScalar(t, config.Parameters[1])
  pseudocount := NewScalar(t, config.Parameters[2])

  if tmp, err := NewGammaDistribution(alpha, beta, pseudocount); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *GammaDistribution) Export(writer io.Writer) error {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  config := NewConfigDistribution("gamma distribution", parameters)

  return config.Export(writer)
}
