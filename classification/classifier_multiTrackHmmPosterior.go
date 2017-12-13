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

package classification

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/ngstat/statistics/matrixDistribution"

/* -------------------------------------------------------------------------- */

type MultiTrackHmmPosterior struct {
  *Hmm
  States   []int
  LogScale   bool
}

/* -------------------------------------------------------------------------- */

func (obj MultiTrackHmmPosterior) CloneMultiTrackClassifier() MultiTrackClassifier {
  return MultiTrackHmmPosterior{obj.Clone(), obj.States, obj.LogScale}
}

/* -------------------------------------------------------------------------- */

func (obj MultiTrackHmmPosterior) Dims() (int, int) {
  return obj.Hmm.Dims()
}

func (obj MultiTrackHmmPosterior) Eval(r Vector, x Matrix) error {
  m, _ := x.Dims()
  if r.Dim() != m {
    return fmt.Errorf("r has invalid length")
  }
  if p, err := obj.PosteriorMarginals(x); err != nil {
    return err
  } else {
    t := NewBareReal(0.0)
    for i := 0; i < m; i++ {
      r.At(i).SetValue(math.Inf(-1))
      for j := 0; j < len(obj.States); j++ {
        r.At(i).LogAdd(r.At(i), p[obj.States[j]].At(i), t)
      }
      if !obj.LogScale {
        r.At(i).Exp(r.At(i))
      }
    }
  }
  return nil
}

func (obj MultiTrackHmmPosterior) Transposed() bool {
  return true
}
