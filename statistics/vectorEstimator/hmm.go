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
import   "github.com/pbenner/ngstat/statistics/generic"
import   "github.com/pbenner/ngstat/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type HmmEstimator struct {
  hmm1        *vectorDistribution.Hmm
  hmm2        *vectorDistribution.Hmm
  hmm3        *vectorDistribution.Hmm
  data         HmmDataSet
  estimators []ScalarEstimator
  // Baum-Welch arguments
  epsilon      float64
  maxSteps     int
  args       []interface{}
}

func NewHmmEstimator(hmm *vectorDistribution.Hmm, estimators []ScalarEstimator, epsilon float64, maxSteps int, args... interface{}) (*HmmEstimator, error) {
  if hmm.NEDists() > 0 && len(estimators) != hmm.NEDists() {
    return nil, fmt.Errorf("invalid number of estimators")
  }
  for i, estimator := range estimators {
    // initialize distribution
    if hmm.Edist[i] == nil {
      hmm.Edist[i] = estimator.GetEstimate()
    }
  }
  // initialize estimators with data
  r := HmmEstimator{}
  r.hmm1       = hmm.Clone()
  r.hmm2       = hmm.Clone()
  r.hmm3       = hmm.Clone()
  r.estimators = estimators
  r.epsilon    = epsilon
  r.maxSteps   = maxSteps
  r.args       = args
  return &r, nil
}

/* Baum-Welch interface
 * -------------------------------------------------------------------------- */

func (obj *HmmEstimator) GetBasicHmm() generic.BasicHmm {
  return obj.hmm1
}

func (obj *HmmEstimator) EvaluateLogPdf(pool ThreadPool) error {
  return obj.data.EvaluateLogPdf(obj.hmm2.Edist, pool)
}

func (obj *HmmEstimator) Swap() {
  obj.hmm1, obj.hmm2, obj.hmm3 = obj.hmm3, obj.hmm1, obj.hmm2
}

func (obj *HmmEstimator) Emissions(gamma []DenseBareRealVector, p ThreadPool) error {
  hmm1 := obj.hmm1
  hmm2 := obj.hmm2
  // estimate emission parameters
  g := p.NewJobGroup()
  if err := p.AddRangeJob(0, len(hmm1.Edist), g, func(c int, p ThreadPool, erf func() error) error {
    // copy parameters for faster convergence
    p1 := hmm1.Edist[c].GetParameters()
    p2 := hmm2.Edist[c].GetParameters()
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
    if err := hmm1.Edist[c].SetParameters(obj.estimators[c].GetParameters()); err != nil {
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

func (obj *HmmEstimator) Step(meta DenseBareRealVector, tmp []generic.BaumWelchTmp, p ThreadPool) (float64, error) {
  hmm1 := obj.hmm1
  hmm2 := obj.hmm2
  return hmm1.Hmm.BaumWelchStep(&hmm1.Hmm, &hmm2.Hmm, obj.data, meta, tmp, p)
}

/* estimator interface
 * -------------------------------------------------------------------------- */

func (obj *HmmEstimator) CloneVectorEstimator() VectorEstimator {
  estimators := make([]ScalarEstimator, len(obj.estimators))
  for i := 0; i < len(obj.estimators); i++ {
    estimators[i] = obj.estimators[i].CloneScalarEstimator()
  }
  r := HmmEstimator{}
  r  = *obj
  r.hmm1       = r.hmm1.Clone()
  r.hmm2       = r.hmm2.Clone()
  r.hmm3       = r.hmm3.Clone()
  r.estimators = estimators
  return &r
}

func (obj *HmmEstimator) ScalarType() ScalarType {
  return obj.hmm1.ScalarType()
}

func (obj *HmmEstimator) GetParameters() Vector {
  return obj.hmm1.GetParameters()
}

func (obj *HmmEstimator) SetParameters(parameters Vector) error {
  return obj.hmm1.SetParameters(parameters)
}

func (obj *HmmEstimator) SetData(x []Vector, n int) error {
  if data, err := NewHmmStdDataSet(obj.ScalarType(), x, obj.hmm1.NEDists()); err != nil {
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

func (obj *HmmEstimator) Estimate(gamma DenseBareRealVector, p ThreadPool) error {
  data     := obj.data
  nRecords := data.GetNRecords()
  nMapped  := data.GetNMapped()
  nData    := 0
  // determine length of the longest sequence
  for i := 0; i < data.GetNRecords(); i++ {
    r := data.GetRecord(i)
    if r.GetN() > nData {
      nData = r.GetN()
    }
  }
  return generic.BaumWelchAlgorithm(obj, gamma, nRecords, nData, nMapped, obj.hmm1.NStates(), obj.hmm1.NEDists(), obj.epsilon, obj.maxSteps, p, obj.args...)
}

func (obj *HmmEstimator) EstimateOnData(x []Vector, gamma DenseBareRealVector, p ThreadPool) error {
  if err := obj.SetData(x, len(x)); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *HmmEstimator) GetEstimate() VectorDistribution {
  return obj.hmm1
}
