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

import   "fmt"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/generic"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type MixtureEstimator struct {
  mixture1    *scalarDistribution.Mixture
  mixture2    *scalarDistribution.Mixture
  mixture3    *scalarDistribution.Mixture
  data         MixtureDataSet
  estimators []ScalarEstimator
  // EM arguments
  epsilon      float64
  maxSteps     int
  args       []interface{}
}

func NewMixtureEstimator(mixture *scalarDistribution.Mixture, estimators []ScalarEstimator, epsilon float64, maxSteps int, args... interface{}) (*MixtureEstimator, error) {
  if len(estimators) != mixture.NComponents() {
    return nil, fmt.Errorf("invalid number of estimators")
  }
  for i, estimator := range estimators {
    // initialize distribution
    if mixture.Edist[i] == nil {
      mixture.Edist[i] = estimator.GetEstimate()
    }
  }
  // initialize estimators with data
  r := MixtureEstimator{}
  r.mixture1   = mixture.Clone()
  r.mixture2   = mixture.Clone()
  r.mixture3   = mixture.Clone()
  r.estimators = estimators
  r.epsilon    = epsilon
  r.maxSteps   = maxSteps
  r.args       = args
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *MixtureEstimator) GetBasicMixture() generic.BasicMixture {
  return obj.mixture1
}

func (obj *MixtureEstimator) EvaluateLogPdf(pool ThreadPool) error {
  return obj.data.EvaluateLogPdf(obj.mixture2.Edist, pool)
}

func (obj *MixtureEstimator) Swap() {
  obj.mixture1, obj.mixture2, obj.mixture3 = obj.mixture3, obj.mixture1, obj.mixture2
}

func (obj *MixtureEstimator) Emissions(gamma []DenseBareRealVector, p ThreadPool) error {
  mixture1 := obj.mixture1
  mixture2 := obj.mixture2
  // estimate emission parameters
  g := p.NewJobGroup()
  if err := p.AddRangeJob(0, mixture1.NComponents(), g, func(c int, p ThreadPool, erf func() error) error {
    // copy parameters for faster convergence
    p1 := mixture1.Edist[c].GetParameters()
    p2 := mixture2.Edist[c].GetParameters()
    for j := 0; j < p1.Dim(); j++ {
      p1.At(j).Set(p2.At(j))
    }
    if err := obj.estimators[c].SetParameters(p1); err != nil {
      return err
    }
    // estimate parameters of the emission distribution
    if err := obj.estimators[c].Estimate(gamma[c], p); err != nil {
      return err
    }
    // update emission distribution
    if err := mixture1.Edist[c].SetParameters(obj.estimators[c].GetParameters()); err != nil {
      return err
    }
    return nil
  }); err != nil {
    return err
  }
  if err := p.Wait(g); err != nil {
    return err
  }
  return nil
}

func (obj *MixtureEstimator) Step(gamma DenseBareRealVector, tmp []generic.EmTmp, p ThreadPool) (float64, error) {
  mixture1 := obj.mixture1
  mixture2 := obj.mixture2
  return mixture1.Mixture.EmStep(&mixture1.Mixture, &mixture2.Mixture, obj.data, gamma, tmp, p)
}

/* -------------------------------------------------------------------------- */

func (obj *MixtureEstimator) CloneScalarEstimator() ScalarEstimator {
  estimators := make([]ScalarEstimator, len(obj.estimators))
  for i := 0; i < len(obj.estimators); i++ {
    estimators[i] = obj.estimators[i].CloneScalarEstimator()
  }
  r := MixtureEstimator{}
  r  = *obj
  r.mixture1   = r.mixture1.Clone()
  r.mixture2   = r.mixture2.Clone()
  r.mixture3   = r.mixture3.Clone()
  r.estimators = estimators
  return &r
}

func (obj *MixtureEstimator) ScalarType() ScalarType {
  return obj.mixture1.ScalarType()
}

func (obj *MixtureEstimator) GetParameters() Vector {
  return obj.mixture1.GetParameters()
}

func (obj *MixtureEstimator) SetParameters(parameters Vector) error {
  return obj.mixture1.SetParameters(parameters)
}

func (obj *MixtureEstimator) SetData(x Vector, n int) error {
  if data, err := NewStdMixtureDataSet(obj.ScalarType(), x, obj.mixture1.NComponents()); err != nil {
    return err
  } else {
    for _, estimator := range obj.estimators {
      // set data
      if err := estimator.SetData(data.GetMappedData(), n); err != nil {
        return err
      }
    }
    obj.data = data
  }
  return nil
}

func (obj *MixtureEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  data     := obj.data
  nMapped  := data.GetNMapped()
  nData    := data.GetN()
  return generic.EmAlgorithm(obj, gamma, nData, nMapped, obj.mixture1.NComponents(), obj.epsilon, obj.maxSteps, p, obj.args...)
}

func (obj *MixtureEstimator) EstimateOnData(x Vector, gamma DenseBareRealVector, p ThreadPool) error {
  if err := obj.SetData(x, x.Dim()); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *MixtureEstimator) GetEstimate() ScalarDistribution {
  return obj.mixture1
}
