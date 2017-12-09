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
import   "math"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/generic"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type HmmDataSet interface {
  generic.AbstractDataSet
  GetMappedData () []Scalar
  EvaluateLogPdf(edist []ScalarDistribution, pool ThreadPool) error
}

/* -------------------------------------------------------------------------- */

type HmmDataRecord struct {
  index []int
  p       Matrix
}

func (obj HmmDataRecord) MapIndex(k int) int {
  return obj.index[k]
}

func (obj HmmDataRecord) GetN() int {
  return len(obj.index)
}

func (obj HmmDataRecord) LogPdf(r Scalar, c, k int) error {
  r.Set(obj.p.At(c, obj.MapIndex(k)))
  return nil
}

/* -------------------------------------------------------------------------- */

type StdHmmDataSet struct {
  // vector of unique observations
  values   []Scalar
  index  [][]int
  // matrix with emission probabilities, each row corresponds
  // to an emission distribution and each column to a unique
  // observation
  p Matrix
  // number of observations
  n int
}

func NewStdHmmDataSet(t ScalarType, x []Vector, k int) (*StdHmmDataSet, error) {
  xMap   := make(map[[1]float64]int)
  index  := make([][]int, len(x))
  values := []Scalar{}
  m      := 0
  // convert vector elements to arrays, which can be used
  // as keys for xMap
  datum := [1]float64{0}
  for d := 0; d < len(x); d++ {
    index[d] = make([]int, x[d].Dim())
    for i := 0; i < x[d].Dim(); i++ {
      datum[0] = x[d].At(i).GetValue()
      if idx, ok := xMap[datum]; ok {
        index [d][i]  = idx
      } else {
        idx   := len(values)
        values = append(values, x[d].At(i))
        xMap [datum] = idx
        index[d][i]  = idx
      }
    }
    m += x[d].Dim()
  }
  r := StdHmmDataSet{}
  r.values = values
  r.index  = index
  r.p      = NullMatrix(t, k, len(values))
  r.n      = m
  return &r, nil
}

func (obj *StdHmmDataSet) GetMappedData() []Scalar {
  return obj.values
}

func (obj *StdHmmDataSet) GetRecord(i int) generic.AbstractDataRecord {
  return HmmDataRecord{obj.index[i], obj.p}
}

func (obj *StdHmmDataSet) GetNMapped() int {
  return len(obj.values)
}

func (obj *StdHmmDataSet) GetNRecords() int {
  return len(obj.index)
}

func (obj *StdHmmDataSet) GetN() int {
  return obj.n
}

func (obj *StdHmmDataSet) EvaluateLogPdf(edist []ScalarDistribution, pool ThreadPool) error {
  x    := obj.values
  p    := obj.p
  m, n := obj.p.Dims()
  if len(edist) != m {
    return fmt.Errorf("data has invalid dimension")
  }
  // distributions may have state and must be cloned
  // for each thread
  d := make([][]ScalarDistribution, pool.NumberOfThreads())
  s := make([]float64,              pool.NumberOfThreads())
  for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
    d[threadIdx] = make([]ScalarDistribution, m)
    for j := 0; j < m; j++ {
      d[threadIdx][j] = edist[j].CloneScalarDistribution()
    }
  }
  g := pool.NewJobGroup()
  // evaluate emission distributions
  if err := pool.AddRangeJob(0, n, g, func(i int, pool ThreadPool, erf func() error) error {
    if erf() != nil {
      return nil
    }
    s[pool.GetThreadId()] = math.Inf(-1)
    // loop over emission distributions
    for j := 0; j < m; j++ {
      if err := d[pool.GetThreadId()][j].LogPdf(p.At(j, i), x[i]); err != nil {
        return err
      }
      s[pool.GetThreadId()] = LogAdd(s[pool.GetThreadId()], p.At(j, i).GetValue())
    }
    if math.IsInf(s[pool.GetThreadId()], -1) {
      return fmt.Errorf("probability is zero for all models on observation `%v'", x[i])
    }
    return nil
  }); err != nil {
    return fmt.Errorf("evaluating emission probabilities failed: %v", err)
  }
  if err := pool.Wait(g); err != nil {
    return fmt.Errorf("evaluating emission probabilities failed: %v", err)
  }
  return nil
}
