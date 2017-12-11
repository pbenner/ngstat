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

package vectorEstimator

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ScalarIdEstimator struct {
  Estimators []ScalarEstimator
  estimate     VectorDistribution
}

/* -------------------------------------------------------------------------- */

func NewScalarIdEstimator(estimators []ScalarEstimator) (*ScalarIdEstimator, error) {
  for i := 0; i < len(estimators); i++ {
    if estimators[i] == nil {
      return nil, fmt.Errorf("estimator must not be nil")
    }
  }
  r := ScalarIdEstimator{}
  r.Estimators = estimators
  if err := r.updateEstimate(); err != nil {
    return nil, err
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIdEstimator) Clone() *ScalarIdEstimator {
  r := ScalarIdEstimator{}
  r.Estimators = make([]ScalarEstimator, len(obj.Estimators))
  for i, estimator := range obj.Estimators {
    r.Estimators[i] = estimator.CloneScalarEstimator()
  }
  r.updateEstimate()
  return &r
}

func (obj *ScalarIdEstimator) CloneVectorEstimator() VectorEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIdEstimator) ScalarType() ScalarType {
  if len(obj.Estimators) == 0 {
    return nil
  } else {
    return obj.Estimators[0].ScalarType()
  }
}

func (obj *ScalarIdEstimator) Dim() int {
  return len(obj.Estimators)
}

func (obj *ScalarIdEstimator) GetParameters() Vector {
  if len(obj.Estimators) == 0 {
    return nil
  }
  p := obj.Estimators[0].GetParameters()
  for i := 1; i < len(obj.Estimators); i++ {
    p = p.AppendVector(obj.Estimators[i].GetParameters())
  }
  return p
}

func (obj *ScalarIdEstimator) SetParameters(parameters Vector) error {
  for i := 0; i < len(obj.Estimators); i++ {
    n := obj.Estimators[i].GetParameters().Dim()
    if parameters.Dim() < n {
      return fmt.Errorf("invalid set of parameters")
    }
    if err := obj.Estimators[i].SetParameters(parameters.Slice(0,n)); err != nil {
      return err
    }
    parameters = parameters.Slice(n,parameters.Dim())
  }
  if parameters.Dim() != 0 {
    return fmt.Errorf("invalid set of parameters")
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarIdEstimator) SetData(x Matrix, n int) error {
  if x == nil {
    for _, estimator := range obj.Estimators {
      if err := estimator.SetData(nil, n); err != nil {
        return err
      }
    }
  } else {
    _, ncol := x.Dims()

    if ncol != obj.Dim() {
      return fmt.Errorf("data has invalid dimension (expected dimension `%d' but data has dimension `%d')", obj.Dim(), ncol)
    }
    for i, estimator := range obj.Estimators {
      if err := estimator.SetData(x.Col(i), n); err != nil {
        return err
      }
    }
  }
  return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ScalarIdEstimator) updateEstimate() error {
  r := make([]ScalarDistribution, len(obj.Estimators))
  for i, estimator := range obj.Estimators {
    r[i] = estimator.GetEstimate()
  }
  if s, err := vectorDistribution.NewScalarId(r...); err != nil {
    return err
  } else {
    obj.estimate = s
  }
  return nil
}

func (obj *ScalarIdEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  for _, estimator := range obj.Estimators {
    if err := estimator.Estimate(gamma, p); err != nil {
      return err
    }
  }
  if err := obj.updateEstimate(); err != nil {
    return err
  }
  return nil
}

func (obj *ScalarIdEstimator) EstimateOnData(x Matrix, gamma DenseBareRealVector, p ThreadPool) error {
  n, _ := x.Dims()

  if err := obj.SetData(x, n); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *ScalarIdEstimator) GetEstimate() VectorDistribution {
  return obj.estimate
}