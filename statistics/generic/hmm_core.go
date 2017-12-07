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
import   "math"

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

type CoreHmm struct {
  Pi          ProbabilityVector // initial probability vector
  Tr          TransitionMatrix  // transition matrix
  Tf          TransitionMatrix  // transition matrix for last transition
  // state equivalence classes define which states share the same
  // emission distribution
  stateMap  []int
  // number of states
  M           int
  // force start and end states
  startStates map[int]bool
  finalStates map[int]bool
}

/* -------------------------------------------------------------------------- */

func newCoreHmm(pi ProbabilityVector, tr TransitionMatrix, stateMap []int, normalize bool) (*CoreHmm, error) {
  r       := CoreHmm{}
  m, _    := tr.Dims()
  r.Pi     = pi
  r.Tr     = tr
  r.Tf     = tr
  r.M      = m
  r.stateMap = stateMap
  if stateMap == nil {
    // generate new state map
    for i := 0; i < r.M; i++ {
      r.stateMap = append(r.stateMap, i)
    }
  } else {
    // clone state map
    r.stateMap = make([]int, len(stateMap))
    copy(r.stateMap, stateMap)
  }
  if normalize {
    // normalize transition matrix and Pi
    t1 := NullScalar(pi.ElementType())
    t2 := NullScalar(pi.ElementType())
    if err := r.normalize(t1, t2); err != nil {
      return nil, err
    }
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (CoreHmm) checkParameters(pi Vector, tr Matrix, stateMap []int) (int, error) {
  n, _ := tr.Dims()
  k    := n
  if m := pi.Dim(); m != n {
    return 0, fmt.Errorf("parameters have invalid dimension")
  }
  if _, m := tr.Dims(); m != n {
    return 0, fmt.Errorf("transition matrix is not a square matrix")
  }
  if stateMap != nil {
    if len(stateMap) != n {
      return 0, fmt.Errorf("invalid state map")
    }
    // determine maximum state
    k = 0
    for _, s := range stateMap {
      if s < 0 {
        return 0, fmt.Errorf("invalid state map")
      }
      if s+1 > k {
        k = s+1
      }
    }
  }
  return k, nil
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) Clone() *CoreHmm {
  pi := obj.Pi.CloneProbabilityVector()
  tr := obj.Tr.CloneTransitionMatrix()
  r, _ := newCoreHmm(pi, tr, obj.stateMap, false)
  if obj.startStates != nil {
    states := []int{}
    for i, _ := range obj.startStates {
      states = append(states, i)
    }
    r.SetStartStates(states)
  }
  if obj.finalStates != nil {
    states := []int{}
    for i, _ := range obj.finalStates {
      states = append(states, i)
    }
    r.SetFinalStates(states)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) normalizePi(t1, t2 Scalar) error {
  if obj.startStates != nil {
    for i := 0; i < obj.M; i++ {
      if _, ok := obj.startStates[i]; !ok {
        // set probability for every state to zero
        // if it is not a start state
        obj.Pi.At(i).SetValue(math.Inf(-1))
      }
    }
  }
  return obj.Pi.Normalize()
}

func (obj *CoreHmm) normalizeTr(t1, t2 Scalar) error {
  return obj.Tr.Normalize()
}

func (obj *CoreHmm) normalizeTf(t1, t2 Scalar) error {
  if obj.finalStates != nil {
    for i := 0; i < obj.M; i++ {
      for j := 0; j < obj.M; j++ {
        if _, ok := obj.finalStates[j]; !ok {
          // set probability for every state to zero
          // if it is not an end state
          obj.Tf.At(i,j).SetValue(math.Inf(-1))
        } else {
          // copy value from Tr
          obj.Tf.At(i,j).Set(obj.Tr.At(i,j))
        }
      }
    }
    return obj.Tf.Normalize()
  }
  return nil
}

func (obj *CoreHmm) normalize(t1, t2 Scalar) error {
  if err := obj.normalizePi(t1, t2); err != nil {
    return err
  }
  if err := obj.normalizeTr(t1, t2); err != nil {
    return err
  }
  if err := obj.normalizeTf(t1, t2); err != nil {
    return err
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (hmm1 *CoreHmm) distance(hmm2 *CoreHmm) float64 {
  r    := math.Inf(-1)
  for i := 0; i < hmm1.M; i++ {
    if t := math.Abs(math.Exp(hmm1.Pi.At(i).GetValue()) - math.Exp(hmm2.Pi.At(i).GetValue())); t > r {
      r = t
    }
    for j := 0; j < hmm1.M; j++ {
      if t := math.Abs(math.Exp(hmm1.Tr.At(i, j).GetValue()) - math.Exp(hmm2.Tr.At(i, j).GetValue())); t > r {
        r = t
      }
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) SetStartStates(states []int) error {
  for _, i := range states {
    if i < -1 || i >= obj.M {
      return fmt.Errorf("invalid start state")
    }
  }
  obj.startStates = make(map[int]bool)
  for _, i := range states {
    obj.startStates[i] = true
  }
  t1 := NewReal(math.Inf(-1))
  t2 := NewReal(math.Inf(-1))
  obj.normalizePi(t1,t2)
  return nil
}

func (obj *CoreHmm) SetFinalStates(states []int) error {
  for _, i := range states {
    if i < -1 || i >= obj.M {
      return fmt.Errorf("invalid final state")
    }
  }
  obj.finalStates = make(map[int]bool)
  for _, i := range states {
    obj.finalStates[i] = true
  }
  // clone transition matrix and renormalize
  obj.Tf = obj.Tr.CloneTransitionMatrix()
  t1 := NewReal(math.Inf(-1))
  t2 := NewReal(math.Inf(-1))
  obj.normalizeTf(t1, t2)
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) ScalarType() ScalarType {
  return obj.Pi.ElementType()
}

func (obj *CoreHmm) NStates() int {
  return obj.M
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) LogPdf(r Scalar, data AbstractDataRecord) error {
  n := data.GetN()
  m := obj.M
  // test length of x
  if n == 0 {
    r.SetValue(0.0)
    return nil
  }
  // result at time t-1
  alpha_s := NullVector(obj.ScalarType(), m)
  // result at time t
  alpha_t := NullVector(obj.ScalarType(), m)
  // some temporary variables
  t1 := r
  t2 := NullScalar(obj.ScalarType())
  // initialize alpha_s
  for i := 0; i < m; i++ {
    if err := data.LogPdf(t2, obj.stateMap[i], 0); err != nil {
      return err
    }
    alpha_s.At(i).Add(obj.Pi.At(i), t2)
  }
  // loop over x(0), ..., x(N-2)
  for k := 1; k < n-1; k++ {
    // transition to state j
    for j := 0; j < m; j++ {
      alpha_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_{x_i} p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      for i := 0; i < m; i++ {
        t1.Add(obj.Tr.At(i, j), alpha_s.At(i))
        alpha_t.At(j).LogAdd(alpha_t.At(j), t1, t2)
      }
      // alpha_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) alpha_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], k); err != nil {
        return err
      }
      alpha_t.At(j).Add(alpha_t.At(j), t2)
    }
    // swap alpha
    alpha_s, alpha_t = alpha_t, alpha_s
  }
  if n > 1 {
    // last step from x(N-2) to x(N-1)
    // transition to state j
    for j := 0; j < m; j++ {
      alpha_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_{x_i} p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      for i := 0; i < m; i++ {
        t1.Add(obj.Tf.At(i, j), alpha_s.At(i))
        alpha_t.At(j).LogAdd(alpha_t.At(j), t1, t2)
      }
      // alpha_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) alpha_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], n-1); err != nil {
        return err
      }
      alpha_t.At(j).Add(alpha_t.At(j), t2)
    }
    // swap alpha
    alpha_s, alpha_t = alpha_t, alpha_s
  }
  // sum up alpha, which gives the final result
  r.SetValue(math.Inf(-1))
  for j := 0; j < m; j++ {
    r.LogAdd(r, alpha_s.At(j), t2)
  }
  return nil
}

/* -------------------------------------------------------------------------- */

// Compute the posterior:
// P(Y_1 in y_1, Y_2 in y_2, ..., Y_n in y_n | X_1, dots, X_n)
//
func (obj *CoreHmm) Posterior(r Scalar, data AbstractDataRecord, states [][]int) error {
  n := data.GetN()
  m := obj.M
  // test length of x
  if n != len(states) {
    return fmt.Errorf("number of states does not match number of observations")
  }
  if n == 0 {
    r.SetValue(0.0)
    return nil
  }
  // result at time t-1 (restricted to given states)
  alpha_s := NullVector(obj.ScalarType(), m)
  // result at time t   (restricted to given states)
  alpha_t := NullVector(obj.ScalarType(), m)
  // result at time t-1
  beta_s := NullVector(obj.ScalarType(), m)
  // result at time t
  beta_t := NullVector(obj.ScalarType(), m)
  // some temporary variables
  t1 := NullScalar(obj.ScalarType())
  t2 := NullScalar(obj.ScalarType())
  // initialize alpha_s
  for _, i := range states[0] {
    if err := data.LogPdf(t2, obj.stateMap[i], 0); err != nil {
      return err
    }
    alpha_s.At(i).Add(obj.Pi.At(i), t2)
  }
  for i := 0; i < m; i++ {
    if err := data.LogPdf(t2, obj.stateMap[i], 0); err != nil {
      return err
    }
    beta_s.At(i).Add(obj.Pi.At(i), t2)
  }
  // loop over x(0), ..., x(N-2)
  for k := 1; k < n-1; k++ {
    // transition to state j
    for _, j := range states[k] {
      alpha_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_{x_i} p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      for _, i := range states[k-1] {
        t1.Add(obj.Tr.At(i, j), alpha_s.At(i))
        alpha_t.At(j).LogAdd(alpha_t.At(j), t1, t2)
      }
      // alpha_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) alpha_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], k); err != nil {
        return err
      }
      alpha_t.At(j).Add(alpha_t.At(j), t2)
    }
    // swap alpha
    alpha_s, alpha_t = alpha_t, alpha_s
    // transition to state j
    for j := 0; j < m; j++ {
      beta_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // beta_t(x_j) = sum_{x_i} p(x_j | x_i) beta_s(x_i)
      // transitions from state i
      for i := 0; i < m; i++ {
        t1.Add(obj.Tr.At(i, j), beta_s.At(i))
        beta_t.At(j).LogAdd(beta_t.At(j), t1, t2)
      }
      // beta_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) beta_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], k); err != nil {
        return err
      }
      beta_t.At(j).Add(beta_t.At(j), t2)
    }
    // swap beta
    beta_s, beta_t = beta_t, beta_s
  }
  if n > 1 {
    // last step from x(N-2) to x(N-1)
    // transition to state j
    for _, j := range states[n-1] {
      alpha_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_{x_i} p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      for _, i := range states[n-2] {
        t1.Add(obj.Tf.At(i, j), alpha_s.At(i))
        alpha_t.At(j).LogAdd(alpha_t.At(j), t1, t2)
      }
      // alpha_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) alpha_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], n-1); err != nil {
        return err
      }
      alpha_t.At(j).Add(alpha_t.At(j), t2)
    }
    // swap beta
    alpha_s, alpha_t = alpha_t, alpha_s
    // transition to state j
    for j := 0; j < m; j++ {
      beta_t.At(j).SetValue(math.Inf(-1))
      // compute:
      // beta_t(x_j) = sum_{x_i} p(x_j | x_i) beta_s(x_i)
      // transitions from state i
      for i := 0; i < m; i++ {
        t1.Add(obj.Tf.At(i, j), beta_s.At(i))
        beta_t.At(j).LogAdd(beta_t.At(j), t1, t2)
      }
      // beta_t(x_t) = p(y_t | x_t) sum_{x_s} p(x_t | x_s) beta_s(x_s)
      if err := data.LogPdf(t2, obj.stateMap[j], n-1); err != nil {
        return err
      }
      beta_t.At(j).Add(beta_t.At(j), t2)
    }
    // swap beta
    beta_s, beta_t = beta_t, beta_s
  }
  // sum up alpha
  r.SetValue(math.Inf(-1))
  for _, j := range states[n-1] {
    r.LogAdd(r, alpha_s.At(j), t2)
  }
  // sum up beta
  t1.SetValue(math.Inf(-1))
  for j := 0; j < m; j++ {
    t1.LogAdd(t1, beta_s.At(j), t2)
  }
  // compute result
  r.Sub(r, t1)
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) forward(data AbstractDataRecord, alpha Matrix, t1, t2 Scalar, n, m int) (Matrix, error) {
  // initialize first position
  if n > 0 {
    for i := 0; i < m; i++ {
      if err := data.LogPdf(t2, obj.stateMap[i], 0); err != nil {
        return nil, err
      }
      alpha.At(i, 0).Add(obj.Pi.At(i), t2)
    }
  }
  // loop over x(0), ..., x(N-2)
  for k := 1; k < n-1; k++ {
    // transition to state j
    for j := 0; j < m; j++ {
      at := alpha.At(j, k)
      // initialize alpha
      at.SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_i p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      for i := 0; i < m; i++ {
        t1.   Add(obj.Tr.At(i, j), alpha.At(i, k-1))
        at.LogAdd(at, t1, t2)
      }
      // alpha_t(x_j) = p(y_k | x_j) sum_i p(x_j | x_i) alpha_s(x_i)
      if err := data.LogPdf(t2, obj.stateMap[j], k); err != nil {
        return nil, err
      }
      at.Add(at, t2)
    }
  }
  if n > 1 {
    // last step from x(N-2) to x(N-1)
    // transition to state j
    for j := 0; j < m; j++ {
      at := alpha.At(j, n-1)
      // initialize alpha
      at.SetValue(math.Inf(-1))
      // compute:
      // alpha_t(x_j) = sum_i p(x_j | x_i) alpha_s(x_i)
      // transitions from state i
      if n > 1 {
        for i := 0; i < m; i++ {
          t1.   Add(obj.Tf.At(i, j), alpha.At(i, n-2))
          at.LogAdd(at, t1, t2)
        }
      }
      // alpha_t(x_j) = p(y_k | x_j) sum_i p(x_j | x_i) alpha_s(x_i)
      if err := data.LogPdf(t2, obj.stateMap[j], n-1); err != nil {
        return nil, err
      }
      at.Add(at, t2)
    }
  }
  return alpha, nil
}

func (obj *CoreHmm) backward(data AbstractDataRecord, beta Matrix, t1, t2 Scalar, n, m int) (Matrix, error) {
  // initialize last position
  if n > 0 {
    for i := 0; i < m; i++ {
      beta.At(i, n-1).SetValue(0.0)
    }
  }
  if n > 1 {
    // first step from x(N-1) to x(N-2)
    // transitions from state i
    for i := 0; i < m; i++ {
      bs := beta.At(i, n-2)
      // initialize beta
      bs.SetValue(math.Inf(-1))
      // compute:
      // beta_s(x_i) = sum_j p(y_{k+1} | x_j) p(x_j | x_i) beta_t(x_j)
      // transition to state j
      for j := 0; j < m; j++ {
        if err := data.LogPdf(t2, obj.stateMap[j], n-1); err != nil {
          return nil, err
        }
        t1.   Add(obj.Tf.At(i, j), t2)
        bs.LogAdd(bs, t1, t2)
      }
    }
  }
  // loop over x(N-2), ..., x(0)
  for k := n-3; k >= 0; k-- {
    // transitions from state i
    for i := 0; i < m; i++ {
      bs := beta.At(i, k)
      // initialize beta
      bs.SetValue(math.Inf(-1))
      // compute:
      // beta_s(x_i) = sum_j p(y_{k+1} | x_j) p(x_j | x_i) beta_t(x_j)
      // transition to state j
      for j := 0; j < m; j++ {
        t1.   Add(obj.Tr.At(i, j), beta.At(j, k+1))
        if err := data.LogPdf(t2, obj.stateMap[j], k+1); err != nil {
          return nil, err
        }
        t1.   Add(t1, t2)
        bs.LogAdd(bs, t1, t2)
      }
    }
  }
  return beta, nil
}

func (obj *CoreHmm) forwardBackward(data AbstractDataRecord, alpha, beta Matrix, t1, t2 Scalar) (Matrix, Matrix, error) {
  var err error
  // length of the sequence
  n := data.GetN()
  // number of states
  m := obj.M
  // execute forward and backward algorithms
  alpha, err = obj.forward (data, alpha, t1, t2, n, m); if err != nil {
    return nil, nil, err
  }
  beta,  err = obj.backward(data,  beta, t1, t2, n, m); if err != nil {
    return nil, nil, err
  }
  return alpha, beta, nil
}

func (obj *CoreHmm) ForwardBackward(data AbstractDataRecord) (Matrix, Matrix, error) {
  t := obj.ScalarType()
  // length of the sequence
  n := data.GetN()
  // number of states
  m := obj.M
  // forward and backward probabilities
  alpha := NullMatrix(t, m, n)
  beta  := NullMatrix(t, m, n)
  // allocate memory
  t1 := NewScalar(t, 0.0)
  t2 := NewScalar(t, 0.0)
  return obj.forwardBackward(data, alpha, beta, t1, t2)
}

func (obj *CoreHmm) PosteriorMarginals(data AbstractDataRecord) ([]Vector, error) {
  t := BareRealType
  n := data.GetN()
  m := obj.M
  // allocate memory
  t1 := NewScalar(t, 0.0)
  t2 := NewScalar(t, 0.0)
  alpha := NullMatrix(t, m, n)
  beta  := NullMatrix(t, m, n)
  gamma := make([]Vector, m)
  for c := 0; c < m; c++ {
    gamma[c] = NullVector(t, data.GetN())
  }
  // execute forward-backward algorithm
  obj.forwardBackward(data, alpha, beta, t1, t2)
  // compute marginals
  for k := 0; k < n; k++ {
    // normalization constant
    t1.SetValue(math.Inf(-1))
    for i := 0; i < m; i++ {
      gamma[i].At(k).Add(alpha.At(i, k), beta.At(i, k))
      t1.LogAdd(t1, gamma[i].At(k), t2)
    }
    if math.IsInf(t1.GetValue(), -1) {
      return nil, fmt.Errorf("all paths have zero probability")
    }
    // normalize gamma
    for i := 0; i < m; i++ {
      gamma[i].At(k).Sub(gamma[i].At(k), t1)
    }
  }
  return gamma, nil
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) GetParameters() Vector {
  p := Vector(obj.Pi)
  p  = p.AppendVector(obj.Tr.ToVector())
  return p
}

func (obj *CoreHmm) SetParameters(parameters Vector) error {
  m := obj.M
  obj.Pi.Set(parameters.Slice(0, m)); parameters = parameters.Slice(m, parameters.Dim())
  if obj.Tr == obj.Tf {
    obj.Tr.Set(parameters.Slice(0 ,m*m).ToMatrix(m,m)); parameters = parameters.Slice(m*m, parameters.Dim())
    obj.Tf = obj.Tr
  } else {
    obj.Tr.Set(parameters.Slice(0, m*m).ToMatrix(m,m)); parameters = parameters.Slice(m*m, parameters.Dim())
    obj.Tf = obj.Tr.CloneTransitionMatrix()
    t1 := NewReal(0.0)
    t2 := NewReal(0.0)
    obj.normalizeTf(t1, t2)
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) String() string {
  var buffer bytes.Buffer

  pi := obj.Pi.CloneVector()
  pi.Map(func(x Scalar) { x.Exp(x) })
  tr := obj.Tr.CloneMatrix()
  tr.Map(func(x Scalar) { x.Exp(x) })
  tf := tr

  if obj.Tr.GetMatrix() != obj.Tf.GetMatrix() {
    tf = obj.Tf.CloneMatrix()
    tf.Map(func(x Scalar) { x.Exp(x) })
  }
  fmt.Fprintf(&buffer, "Initial probability vector:\n%s\n", pi)
  fmt.Fprintf(&buffer, "Transition matrix:\n%s\n", tr)
  if tr != tf {
    fmt.Fprintf(&buffer, "Final transition matrix:\n%s\n", tf)
  }
  return buffer.String()
}

/* -------------------------------------------------------------------------- */

func (obj *CoreHmm) ImportConfig(config ConfigDistribution, t ScalarType) error {

  pi, ok := config.GetNamedParametersAsFloats("Pi"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  tr, ok := config.GetNamedParametersAsFloats("Tr"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  n, ok := config.GetNamedParameterAsInt("N"); if ! ok {
    return fmt.Errorf("invalid config file")
  }
  stateMap, ok := config.GetNamedParametersAsInts("StateMap"); if ! ok {
    return fmt.Errorf("invalid config file")
  }

  Pi, err := NewHmmProbabilityVector(NewVector(t, pi)); if err != nil {
    return err
  }
  Tr, err := NewHmmTransitionMatrix(NewMatrix(t, n, n, tr)); if err != nil {
    return err
  }

  if tmp, err := newCoreHmm(Pi, Tr, stateMap, true); err != nil {
    return err
  } else {
    *obj = *tmp
  }

  return nil
}

func (obj *CoreHmm) ExportConfig() ConfigDistribution {

  n := obj.Pi.Dim()

  config := struct{
    Pi       []float64
    Tr       []float64
    StateMap []int
    N          int }{}
  config.Pi       = make([]float64, n)
  config.Tr       = make([]float64, n*n)
  config.StateMap = obj.stateMap
  config.N        = n

  for i := 0; i < n; i++ {
    config.Pi[i] = math.Exp(obj.Pi.At(i).GetValue())
    for j := 0; j < n; j++ {
      config.Tr[i*n+j] = math.Exp(obj.Tr.At(i,j).GetValue())
    }
  }

  return NewConfigDistribution("hmm core", config)
}
