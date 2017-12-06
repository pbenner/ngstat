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

package statistics

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "reflect"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type BasicDistribution interface {
  ImportConfig(config ConfigDistribution, t ScalarType) error
  ExportConfig() ConfigDistribution
  GetParameters() Vector
  SetParameters(parameters Vector) error
  ScalarType() ScalarType
}

/* -------------------------------------------------------------------------- */

type ScalarDistribution interface {
  BasicDistribution
  LogPdf(r Scalar, x Scalar) error
  CloneScalarDistribution() ScalarDistribution
}

type VectorDistribution interface {
  BasicDistribution
  LogPdf(r Scalar, x Vector) error
  Dim() int
  CloneVectorDistribution() VectorDistribution
}

type MatrixDistribution interface {
  BasicDistribution
  LogPdf(r Scalar, x Matrix) error
  Dims() (int, int)
  CloneMatrixDistribution() MatrixDistribution
}

/* -------------------------------------------------------------------------- */

var ScalarDistributionRegistry map[string]ScalarDistribution
var VectorDistributionRegistry map[string]VectorDistribution
var MatrixDistributionRegistry map[string]MatrixDistribution

func init() {
  ScalarDistributionRegistry = make(map[string]ScalarDistribution)
  VectorDistributionRegistry = make(map[string]VectorDistribution)
  MatrixDistributionRegistry = make(map[string]MatrixDistribution)
}

/* -------------------------------------------------------------------------- */

func NewScalarDistribution(name string) ScalarDistribution {
  if x, ok := ScalarDistributionRegistry[name]; ok {
    return reflect.New(reflect.TypeOf(x).Elem()).Interface().(ScalarDistribution)
  } else {
    return nil
  }
}

func NewVectorDistribution(name string) VectorDistribution {
  if x, ok := VectorDistributionRegistry[name]; ok {
    return reflect.New(reflect.TypeOf(x).Elem()).Interface().(VectorDistribution)
  } else {
    return nil
  }
}

func NewMatrixDistribution(name string) MatrixDistribution {
  if x, ok := MatrixDistributionRegistry[name]; ok {
    return reflect.New(reflect.TypeOf(x).Elem()).Interface().(MatrixDistribution)
  } else {
    return nil
  }
}
