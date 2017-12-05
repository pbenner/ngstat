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

import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/statistics/config"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/distribution"

/* -------------------------------------------------------------------------- */

type BetaDistribution struct {
  distribution.BetaDistribution
}

/* -------------------------------------------------------------------------- */

func NewBetaDistribution(alpha, beta Scalar, logScale bool) (*BetaDistribution, error) {
  if dist, err := distribution.NewBetaDistribution(alpha, beta, logScale); err != nil {
    return nil, err
  } else {
    return &BetaDistribution{*dist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) Clone() *BetaDistribution {
  return &BetaDistribution{*dist.BetaDistribution.Clone()}
}

func (dist *BetaDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *BetaDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "beta distribution"); err != nil {
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

  alpha    := NewScalar(t, config.Parameters[0])
  beta     := NewScalar(t, config.Parameters[1])
  logScale := config.Parameters[2] == 1.0

  if tmp, err := NewBetaDistribution(alpha, beta, logScale); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *BetaDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("beta distribution", dist.GetParameters())

  return config.Export(writer)
}
