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

package statistics

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "os"
import   "math"
import   "testing"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/vectorDistribution"
import   "github.com/pbenner/ngstat/statistics/scalarEstimator"

import . "github.com/pbenner/autodiff"

import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func TestHmm1(t *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  pi := NewVector(RealType, []float64{0.6, 0.4})
  tr := NewMatrix(RealType, 2, 2,
    []float64{0.7, 0.3, 0.4, 0.6})

  e1, _ := scalarEstimator.NewCategoricalEstimator(
    NewVector(RealType, []float64{0.1, 0.9}))
  e2, _ := scalarEstimator.NewCategoricalEstimator(
    NewVector(RealType, []float64{0.7, 0.3}))

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, nil)
  if err != nil {
    t.Error(err)
  }
  // test Baum-Welch algorithm
  //////////////////////////////////////////////////////////////////////////////
  if estimator, err := NewHmmEstimator(hmm, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
    t.Error(err)
  } else {
    x := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})

    estimator.EstimateOnData([]Vector{x}, nil, ThreadPool{})

    hmm1 := hmm
    hmm2 := estimator.GetEstimate()

    p1 := NullReal(); hmm1.LogPdf(p1, x)
    p2 := NullReal(); hmm2.LogPdf(p2, x)

    if p1.Greater(p2) {
        t.Errorf("Baum-Welch test failed")
    }
    if math.Abs(p2.GetValue() - -4.493268e+00) > 1e-4 {
      t.Errorf("Baum-Welch test failed")
    }
  }
  // // test Baum-Welch algorithm with conditioning
  // //////////////////////////////////////////////////////////////////////////////
  // {
  //   x  := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})

  //   hmm := hmm.Clone()
  //   hmm.SetStartStates([]int{0})
  //   hmm.SetFinalStates([]int{0})

  //   p1 := NullReal(); hmm.LogPdf(p1, x)
  //   hmm, _ = hmm.BaumWelchSV([]Vector{x}, []ScalarEstimator{e1, e2}, 1e-8, -1)
  //   p2 := NullReal(); hmm.LogPdf(p2, x)
  //   if p1.Greater(p2) {
  //     t.Errorf("Baum-Welch test failed")
  //   }
  //   if math.Abs(p2.GetValue() - -5.834855e+00) > 1e-4 {
  //     t.Errorf("Baum-Welch test failed")
  //   }
  // }
}
