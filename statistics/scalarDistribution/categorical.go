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

type CategoricalDistribution struct {
  distribution.CategoricalDistribution
}

/* -------------------------------------------------------------------------- */

func NewCategoricalDistribution(theta Vector) (*CategoricalDistribution, error) {
  if dist, err := distribution.NewCategoricalDistribution(theta); err != nil {
    return nil, err
  } else {
    return &CategoricalDistribution{*dist}, err
  }
}

/* -------------------------------------------------------------------------- */

func (dist *CategoricalDistribution) Clone() *CategoricalDistribution {
  return &CategoricalDistribution{*dist.CategoricalDistribution.Clone()}
}

func (dist *CategoricalDistribution) CloneScalarDistribution() ScalarDistribution {
  return dist.Clone()
}

/* -------------------------------------------------------------------------- */

func (dist *CategoricalDistribution) Import(reader io.Reader, args... interface{}) error {

  var config ConfigDistribution

  if err := config.Import(reader, "categorical distribution"); err != nil {
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

  theta := NewVector(t, config.Parameters)

  if tmp, err := NewCategoricalDistribution(theta); err != nil {
    return err
  } else {
    *dist = *tmp
  }
  return nil
}

func (dist *CategoricalDistribution) Export(writer io.Writer) error {

  config := NewConfigDistribution("categorical distribution", dist.GetParameters())

  return config.Export(writer)
}
