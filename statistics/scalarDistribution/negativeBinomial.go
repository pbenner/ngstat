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

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/distribution"

/* -------------------------------------------------------------------------- */

type NegativeBinomialDistribution struct {
  distribution.NegativeBinomialDistribution
  Pseudocount Scalar
  // state
  x Scalar
}

/* -------------------------------------------------------------------------- */

func NewNegativeBinomialDistribution(r, p, pseudocount Scalar) (*NegativeBinomialDistribution, error) {
  if dist, err := distribution.NewNegativeBinomialDistribution(r, p); err != nil {
    return nil, err
  } else {
    return &NegativeBinomialDistribution{*dist, pseudocount.CloneScalar(), NullScalar(pseudocount.Type())}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) Clone() *NegativeBinomialDistribution {
  return &NegativeBinomialDistribution{
    *dist.NegativeBinomialDistribution.Clone(),
     dist.Pseudocount                 .CloneScalar(),
     dist.x                           .CloneScalar()}
}

func (dist *NegativeBinomialDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) LogPdf(r Scalar, x Scalar) error {
  dist.x.Add(x, dist.Pseudocount)
  return dist.NegativeBinomialDistribution.LogPdf(r, dist.x)
}

/* -------------------------------------------------------------------------- */

func (dist *NegativeBinomialDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  r := NewScalar(t, config.Parameters[0])
  p := NewScalar(t, config.Parameters[1])
  c := NewScalar(t, config.Parameters[2])

  if tmp, err := NewNegativeBinomialDistribution(r, p, c); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *NegativeBinomialDistribution) ExportConfig() ConfigDistribution {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  return NewConfigDistribution("negative binomial distribution", parameters)
}
