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

type LogCauchyDistribution struct {
  distribution.LogCauchyDistribution
  Pseudocount Scalar
  x           Scalar
}

/* -------------------------------------------------------------------------- */

func NewLogCauchyDistribution(mu, sigma, pseudocount Scalar) (*LogCauchyDistribution, error) {
  if dist, err := distribution.NewLogCauchyDistribution(mu, sigma); err != nil {
    return nil, err
  } else {
    return &LogCauchyDistribution{*dist, pseudocount, NullScalar(mu.Type())}, err
  }
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) Clone() *LogCauchyDistribution {
  return &LogCauchyDistribution{*dist.LogCauchyDistribution.Clone(), dist.Pseudocount.CloneScalar(), dist.x.CloneScalar()}
}

func (dist *LogCauchyDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) LogPdf(r Scalar, x Scalar) error {
  dist.x.Add(x, dist.Pseudocount)
  return dist.LogCauchyDistribution.LogPdf(r, dist.x)
}

func (dist *LogCauchyDistribution) Pdf(r Scalar, x Scalar) error {
  if err := dist.LogPdf(r, x); err != nil {
    return err
  }
  r.Exp(r)
  return nil
}

/* -------------------------------------------------------------------------- */

func (dist *LogCauchyDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  mu          := NewScalar(t, config.Parameters[0])
  sigma       := NewScalar(t, config.Parameters[1])
  pseudocount := NewScalar(t, config.Parameters[2])

  if tmp, err := NewLogCauchyDistribution(mu, sigma, pseudocount); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *LogCauchyDistribution) ExportConfig() ConfigDistribution {

  parameters := dist.GetParameters()
  parameters  = parameters.AppendScalar(dist.Pseudocount)

  return NewConfigDistribution("log cauchy distribution", parameters)
}
