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

package classification

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/utility"

/* -------------------------------------------------------------------------- */

func Performance(groundtruth []int, test []float64, n int) ([]float64, []int, []int, []int, []int) {
  n1 := len(groundtruth)
  n2 := len(test)
  if n1 != n2 {
    panic("invalid arguments")
  }
  min := SliceMin(test)
  max := SliceMax(test)
  step := (max-min)/float64(n)
  // result
  thr := []float64{}
  tpv := []int{}
  fpv := []int{}
  tnv := []int{}
  fnv := []int{}
  for t := min; t < max; t += step {
    tp := 0
    tn := 0
    fn := 0
    fp := 0
    for i := 0; i < n1; i++ {
      if groundtruth[i] == 1 && test[i] >= t {
        tp++
      } else if groundtruth[i] == 0 && test[i] >= t {
        fp++
      } else if groundtruth[i] == 0 && test[i] < t {
        tn++
      } else if groundtruth[i] == 1 && test[i] < t {
        fn++
      }
    }
    thr = append(thr, t)
    tpv = append(tpv, tp)
    fpv = append(fpv, fp)
    tnv = append(tnv, tn)
    fnv = append(fnv, fn)
  }
  return thr, tpv, fpv, tnv, fnv
}

func RocCurve(groundtruth []int, test []float64, n int) ([]float64, []float64, []float64) {
  n1 := len(groundtruth)
  n2 := len(test)
  if n1 != n2 {
    panic("invalid arguments")
  }
  min := SliceMin(test)
  max := SliceMax(test)
  step := (max-min)/float64(n)
  // result
  tpr := []float64{}
  fpr := []float64{}
  thr := []float64{}
  for t := min; t < max; t += step {
    tp := 0
    tn := 0
    fn := 0
    fp := 0
    for i := 0; i < n1; i++ {
      if groundtruth[i] == 1 && test[i] >= t {
        tp++
      } else if groundtruth[i] == 0 && test[i] >= t {
        fp++
      } else if groundtruth[i] == 0 && test[i] < t {
        tn++
      } else if groundtruth[i] == 1 && test[i] < t {
        fn++
      }
    }
    tpr = append(tpr, float64(tp)/float64(tp + fn))
    fpr = append(fpr, float64(fp)/float64(fp + tn))
    thr = append(thr, t)
  }
  return thr, fpr, tpr
}

func PrecisionRecallCurve(groundtruth []int, test []float64, n int) ([]float64, []float64, []float64) {
  n1 := len(groundtruth)
  n2 := len(test)
  if n1 != n2 {
    panic("invalid arguments")
  }
  min := SliceMin(test)
  max := SliceMax(test)
  step := (max-min)/float64(n)
  // result
  tpr := []float64{}
  ppv := []float64{}
  thr := []float64{}
  for t := min; t < max; t += step {
    tp := 0
    fn := 0
    fp := 0
    for i := 0; i < n1; i++ {
      if groundtruth[i] == 1 && test[i] >= t {
        tp++
      } else if groundtruth[i] == 0 && test[i] >= t {
        fp++
      } else if groundtruth[i] == 1 && test[i] < t {
        fn++
      }
    }
    tpr = append(tpr, float64(tp)/float64(tp + fn))
    ppv = append(ppv, float64(tp)/float64(tp + fp))
    thr = append(thr, t)
  }
  return thr, tpr, ppv
}

func AUC(x, y []float64) float64 {
  n1 := len(x)
  n2 := len(y)
  if n1 != n2 {
    panic("invalid arguments")
  }
  result := 0.0

  for i := 1; i < n1; i++ {
    dx := math.Abs(x[i] - x[i-1])
    dy := (y[i] + y[i-1])/2.0
    result += dx*dy
  }
  return result
}
