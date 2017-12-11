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

type ScalarBatchIdEstimator struct {
  Estimators []ScalarBatchEstimator
  estimate     VectorDistribution
}

/* -------------------------------------------------------------------------- */

func NewScalarBatchIdEstimator(estimators []ScalarBatchEstimator) (*ScalarBatchIdEstimator, error) {
  for i := 0; i < len(estimators); i++ {
    if estimators[i] == nil {
      return nil, fmt.Errorf("estimator must not be nil")
    }
  }
  r := ScalarBatchIdEstimator{}
  r.Estimators = estimators
  if err := r.updateEstimate(); err != nil {
    return nil, err
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarBatchIdEstimator) Clone() *ScalarBatchIdEstimator {
  r := ScalarBatchIdEstimator{}
  r.Estimators = make([]ScalarBatchEstimator, len(obj.Estimators))
  for i, estimator := range obj.Estimators {
    r.Estimators[i] = estimator.CloneScalarBatchEstimator()
  }
  r.updateEstimate()
  return &r
}

func (obj *ScalarBatchIdEstimator) CloneVectorBatchEstimator() VectorBatchEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *ScalarBatchIdEstimator) ScalarType() ScalarType {
  if len(obj.Estimators) == 0 {
    return nil
  } else {
    return obj.Estimators[0].ScalarType()
  }
}

func (obj *ScalarBatchIdEstimator) Dim() int {
  return len(obj.Estimators)
}

func (obj *ScalarBatchIdEstimator) GetParameters() Vector {
  if len(obj.Estimators) == 0 {
    return nil
  }
  p := obj.Estimators[0].GetParameters()
  for i := 1; i < len(obj.Estimators); i++ {
    p = p.AppendVector(obj.Estimators[i].GetParameters())
  }
  return p
}

func (obj *ScalarBatchIdEstimator) SetParameters(parameters Vector) error {
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

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ScalarBatchIdEstimator) Initialize(p ThreadPool) error {
  for _, estimator := range obj.Estimators {
    if err := estimator.Initialize(p); err != nil {
      return err
    }
  }
  return nil
}

func (obj *ScalarBatchIdEstimator) NewObservation(x Vector, gamma Scalar, p ThreadPool) error {
  if x.Dim() != len(obj.Estimators) {
    return fmt.Errorf("data has invalid dimension")
  }
  for i, estimator := range obj.Estimators {
    if err := estimator.NewObservation(x.At(i), gamma, p); err != nil {
      return err
    }
  }
  return nil
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *ScalarBatchIdEstimator) updateEstimate() error {
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

func (obj *ScalarBatchIdEstimator) GetEstimate() VectorDistribution {
  return obj.estimate
}
