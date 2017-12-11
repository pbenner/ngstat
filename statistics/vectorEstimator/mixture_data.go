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

type MixtureDataSet interface {
  generic.MixtureDataSet
  GetMappedData () Matrix
  EvaluateLogPdf(edist []VectorDistribution, pool ThreadPool) error
}

/* -------------------------------------------------------------------------- */

type StdMixtureDataSet struct {
  values Matrix
  p      Matrix
  n      int
}

func NewStdMixtureDataSet(t ScalarType, x Matrix, k int) (*StdMixtureDataSet, error) {
  n, _ := x.Dims()
  r    := StdMixtureDataSet{}
  r.values = x
  r.p      = NullMatrix(t, k, n)
  r.n      = n
  return &r, nil
}

func (obj *StdMixtureDataSet) MapIndex(k int) int {
  return k
}

func (obj *StdMixtureDataSet) GetMappedData() Matrix {
  return obj.values
}

func (obj *StdMixtureDataSet) GetN() int {
  return obj.n
}

func (obj *StdMixtureDataSet) GetNMapped() int {
  return obj.n
}

func (obj *StdMixtureDataSet) LogPdf(r Scalar, c, i int) error {
  r.Set(obj.p.At(c, i))
  return nil
}

func (obj *StdMixtureDataSet) EvaluateLogPdf(edist []VectorDistribution, pool ThreadPool) error {
  x    := obj.values
  p    := obj.p
  m, n := obj.p.Dims()
  if len(edist) != m {
    return fmt.Errorf("data has invalid dimension")
  }
  // distributions may have state and must be cloned
  // for each thread
  d := make([][]VectorDistribution, pool.NumberOfThreads())
  s := make([]float64, pool.NumberOfThreads())
  for threadIdx := 0; threadIdx < pool.NumberOfThreads(); threadIdx++ {
    d[threadIdx] = make([]VectorDistribution, m)
    for j := 0; j < m; j++ {
      d[threadIdx][j] = edist[j].CloneVectorDistribution()
    }
  }
  g := pool.NewJobGroup()
  // evaluate emission distributions
  if err := pool.AddRangeJob(0, n, g, func(i int, pool ThreadPool, erf func() error) error {
    if erf() != nil {
      return nil
    }
    s := s[pool.GetThreadId()]
    d := d[pool.GetThreadId()]
    s = math.Inf(-1)
    // loop over emission distributions
    for j := 0; j < m; j++ {
      if err := d[j].LogPdf(p.At(j, i), x.Row(i)); err != nil {
        return err
      }
      s = LogAdd(s, p.At(j, i).GetValue())
    }
    if math.IsInf(s, -1) {
      return fmt.Errorf("probability is zero for all models on observation `%v'", x.Row(i))
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
