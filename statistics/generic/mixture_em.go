/* Copyright (C) 2016 Philipp Benner
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
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func (obj *Mixture) EmStep(mixture1, mixture2 *Mixture, data DataSet, gammaMeta DenseBareRealVector, tmp []EmTmp, p ThreadPool) (float64, error) {
  if data.GetNRecords() != 1 {
    return 0, fmt.Errorf("mixture data set is expected to have only one record")
  }
  m  := obj.NComponents()
  r  := data.GetRecord(0)
  g  := p.NewJobGroup()
  // tell every thread that it needs to reset all variables
  for threadIdx := 0; threadIdx < len(tmp); threadIdx++ {
    tmp[threadIdx].init = false
  }
  // compute gamma temporaries
  if err := p.AddRangeJob(0, r.GetN(), g, func(l int, p ThreadPool, erf func() error) error {
    gammaTmp   := tmp[p.GetThreadId()].gammaTmp
    gamma      := tmp[p.GetThreadId()].gamma
    logWeights := tmp[p.GetThreadId()].logWeights
    t1         := tmp[p.GetThreadId()].t1
    t2         := tmp[p.GetThreadId()].t2
    // check init
    if tmp[p.GetThreadId()].init == false {
      logWeights.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
      for i := 0; i < len(gamma); i++ {
        gamma[i].Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
      }
      tmp[p.GetThreadId()].likelihood = 0.0
      tmp[p.GetThreadId()].init       = true
    }
    k := r.MapIndex(l)
    // normalization constant
    t1.SetValue(math.Inf(-1))
    for i := 0; i < m; i++ {
      if err := r.LogPdf(t2, i, k); err != nil {
        return err
      }
      gammaTmp.AT(i).Add(t2, mixture2.LogWeights.At(i))
      t1.LOGADD(t1, gammaTmp.AT(i), t2)
    }
    // update log-likelihood
    tmp[p.GetThreadId()].likelihood += t1.GetValue()
    // normalize gammaTmp
    for i := 0; i < m; i++ {
      gammaTmp.AT(i).Sub(gammaTmp.AT(i), t1)
      if gammaMeta != nil {
        gammaTmp.AT(i).ADD(gammaTmp.AT(i), gammaMeta.AT(k))
      }
    }
    if gamma != nil {
      // add result to gamma
      for i := 0; i < m; i++ {
        gamma[i].AT(k).LOGADD(gamma[i].AT(k), gammaTmp.AT(i), t2)
      }
    }
    if logWeights != nil {
      // add result to weights
      for i := 0; i < m; i++ {
        logWeights.AT(i).LOGADD(logWeights.AT(i), gammaTmp.AT(i), t2)
      }
    }
    return nil
  }); err != nil {
    return math.Inf(-1), nil
  }
  // wait for all threads to finish
  if err := p.Wait(g); err != nil {
    return math.Inf(-1), nil
  }
  if tmp[0].logWeights != nil {
    // set weights to zero
    mixture1.LogWeights.Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
    // compute posterior weights
    for threadIdx := 0; threadIdx < p.NumberOfThreads(); threadIdx++ {
      if tmp[threadIdx].init == false {
        // this thread was never used
        continue
      }
      for i := 0; i < m; i++ {
        mixture1.LogWeights.At(i).LogAdd(mixture1.LogWeights.At(i), tmp[threadIdx].logWeights.At(i), tmp[0].t2)
      }
    }
    // normalize weights
    mixture1.normalize()
  }
  // initialize gamma for thread 0 if necessary
  if tmp[0].init == false {
    for i := 0; i < len(tmp[0].gamma); i++ {
      tmp[0].gamma[i].Map(func(x Scalar) { x.SetValue(math.Inf(-1)) })
    }
    tmp[0].likelihood = 0.0
    tmp[0].init       = true
  }
  // collect gamma results
  for threadIdx := 1; threadIdx < p.NumberOfThreads(); threadIdx++ {
    if tmp[threadIdx].init == false {
      // this thread was never used
      continue
    }
    for i := 0; i < len(tmp[0].gamma); i++ {
      for j := 0; j < tmp[0].gamma[i].Dim(); j++ {
        tmp[0].gamma[i].AT(j).LOGADD(
          tmp[        0].gamma[i].AT(j),
          tmp[threadIdx].gamma[i].AT(j), tmp[0].t2)
      }
    }
    tmp[0].likelihood += tmp[threadIdx].likelihood
  }

  return tmp[0].likelihood, nil
}
