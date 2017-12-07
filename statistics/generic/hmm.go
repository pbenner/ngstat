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

package generic

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "bytes"

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- *
 *
 * The following HMM model is implemented:
 *
 * y(0) -> ... -> y(N-1)
 *   |               |
 *   v               v
 * x(0)    ...    x(N-1)
 *
 * -------------------------------------------------------------------------- */

type Hmm struct {
  CoreHmm
  Edist BasicDistributionSet
}

/* -------------------------------------------------------------------------- */

func NewHmm(pi Vector, tr Matrix, stateMap []int, edist BasicDistributionSet) (*Hmm, error) {
  if _, err := (Hmm{}).checkParameters(pi, tr, stateMap, edist); err != nil {
    return nil, err
  }
  p, err := NewHmmProbabilityVector(pi); if err != nil {
    return nil, err
  }
  t, err := NewHmmTransitionMatrix(tr); if err != nil {
    return nil, err
  }
  return newHmm(p, t, edist, stateMap, true)
}

func newHmm(pi ProbabilityVector, tr TransitionMatrix, edist BasicDistributionSet, stateMap []int, normalize bool) (*Hmm, error) {
  if t, err := newCoreHmm(pi, tr, stateMap, normalize); err != nil {
    return nil, err
  } else {
    return &Hmm{*t, edist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (Hmm) checkParameters(pi Vector, tr Matrix, stateMap []int, edist BasicDistributionSet) (int, error) {
  if k, err := (CoreHmm{}).checkParameters(pi, tr, stateMap); err != nil {
    return 0, err
  } else {
    if edist.Len() > 0 && edist.Len() != k {
      return 0, fmt.Errorf("invalid number of emission distributions")
    }
    return k, nil
  }
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) Clone() *Hmm {
  r := Hmm{}
  r.CoreHmm = *obj.CoreHmm.Clone()
  r.Edist   = obj.Edist.CloneBasicDistributionSet()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) NEDists() int {
  return obj.Edist.Len()
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) GetParameters() Vector {
  p := obj.CoreHmm.GetParameters()
  for i := 0; i < obj.Edist.Len(); i++ {
    p = p.AppendVector(obj.Edist.GetBasicDistribution(i).GetParameters())
  }
  return p
}

func (obj *Hmm) SetParameters(parameters Vector) error {
  n := obj.CoreHmm.GetParameters().Dim()
  obj.CoreHmm.SetParameters(parameters.Slice(0,n))
  parameters  = parameters.Slice(n,parameters.Dim())
  if parameters.Dim() > 0 {
    for i := 0; i < obj.Edist.Len(); i++ {
      n := obj.Edist.GetBasicDistribution(i).GetParameters().Dim()
      if err := obj.Edist.GetBasicDistribution(i).SetParameters(parameters.Slice(0,n)); err != nil {
        return err
      }
      parameters = parameters.Slice(n, parameters.Dim())
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) String() string {
  var buffer bytes.Buffer

  fmt.Fprintf(&buffer, obj.CoreHmm.String())
  fmt.Fprintf(&buffer, "Emissions:\n")
  for i := 0; i < obj.Edist.Len(); i++ {
    fmt.Fprintf(&buffer, "-> %+v\n", obj.Edist.GetBasicDistribution(i).GetParameters())
  }
  return buffer.String()
}
