/* Copyright (C) 2017 Philipp Benner
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

type BasicDistributionSet interface {
  Len                      ()      int
  GetBasicDistribution     (i int) BasicDistribution
  CloneBasicDistributionSet()      BasicDistributionSet
}

/* -------------------------------------------------------------------------- */

type ScalarDistributionSet []ScalarDistribution

func (obj ScalarDistributionSet) Len() int {
  return len(obj)
}

func (obj ScalarDistributionSet) GetBasicDistribution(i int) BasicDistribution {
  return obj[i]
}

func (obj ScalarDistributionSet) CloneBasicDistributionSet() BasicDistributionSet {
  r := make(ScalarDistributionSet, obj.Len())
  for i, d := range obj {
    r[i] = d.CloneScalarDistribution()
  }
  return r
}

/* -------------------------------------------------------------------------- */

type VectorDistributionSet []VectorDistribution

func (obj VectorDistributionSet) Len() int {
  return len(obj)
}

func (obj VectorDistributionSet) GetBasicDistribution(i int) BasicDistribution {
  return obj[i]
}

func (obj VectorDistributionSet) CloneBasicDistributionSet() BasicDistributionSet {
  r := make(VectorDistributionSet, obj.Len())
  for i, d := range obj {
    r[i] = d.CloneVectorDistribution()
  }
  return r
}

/* -------------------------------------------------------------------------- */

type MatrixDistributionSet []MatrixDistribution

func (obj MatrixDistributionSet) Len() int {
  return len(obj)
}

func (obj MatrixDistributionSet) GetBasicDistribution(i int) BasicDistribution {
  return obj[i]
}

func (obj MatrixDistributionSet) CloneBasicDistributionSet() BasicDistributionSet {
  r := make(MatrixDistributionSet, obj.Len())
  for i, d := range obj {
    r[i] = d.CloneMatrixDistribution()
  }
  return r
}
