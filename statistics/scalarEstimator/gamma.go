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

package scalarEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type CategoricalEstimator struct {
  *scalarDistribution.CategoricalDistribution
  StdEstimator
}

func NewCategoricalEstimator(theta Vector) (*CategoricalEstimator, error) {
  if f, err := scalarDistribution.NewCategoricalDistribution(theta); err != nil {
    return nil, err
  } else {
    r := CategoricalEstimator{}
    r.CategoricalDistribution = f
    return &r, nil
  }
}

func (estimator *CategoricalEstimator) CloneScalarEstimator() ScalarEstimator {
  r := CategoricalEstimator{}
  r.CategoricalDistribution = estimator.CategoricalDistribution.Clone()
  r.x = estimator.x
  return &r
}

func (estimator *CategoricalEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  theta := estimator.Theta
  x     := estimator.x
  sum   := NewBareReal(math.Inf(-1))
  tmp   := NewBareReal(0.0)
  // initialize theta
  for i := 0; i < theta.Dim(); i++ {
    theta.At(i).Reset()
    theta.At(i).SetValue(math.Inf(-1))
  }
  // loop over observations
  for k := 0; k < len(x); k++ {
    // discretize observation at position k
    i := int(x[k].GetValue())
    if !math.IsInf(gamma.At(k).GetValue(), -1) {
      theta.At(i).LogAdd(theta.At(i), gamma.At(k), tmp)
    }
  }
  // normalize theta
  for i := 0; i < theta.Dim(); i++ {
    sum.LogAdd(sum, theta.At(i), tmp)
  }
  for i := 0; i < theta.Dim(); i++ {
    theta.At(i).Sub(theta.At(i), sum)
  }
  return nil
}

func (estimator *CategoricalEstimator) EstimateOnData(x []Scalar, gamma DenseBareRealVector, p ThreadPool) error {
  if err := estimator.SetData(x, len(x)); err != nil {
    return err
  }
  return estimator.Estimate(gamma, p)
}

func (estimator *CategoricalEstimator) GetEstimate() ScalarDistribution {
  return estimator.CategoricalDistribution
}
