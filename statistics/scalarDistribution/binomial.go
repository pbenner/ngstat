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

func (dist *BinomialDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "binomial distribution"); err != nil {
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

  theta := NewScalar(t, config.Parameters[0])
  n     := int(config.Parameters[1])

  if tmp, err := NewBinomialDistribution(theta, n); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *BinomialDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("binomial distribution", dist.GetParameters())

  return config.Export(writer)
}
