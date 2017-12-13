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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/ngstat/statistics/matrixDistribution"

/* -------------------------------------------------------------------------- */

type MultiTrackHmmClassifier struct {
  *Hmm
}

/* -------------------------------------------------------------------------- */

func (obj MultiTrackHmmClassifier) CloneMultiTrackClassifier() MultiTrackClassifier {
  return MultiTrackHmmClassifier{obj.Clone()}
}

/* -------------------------------------------------------------------------- */

func (obj MultiTrackHmmClassifier) Dims() (int, int) {
  return obj.Hmm.Dims()
}

func (obj MultiTrackHmmClassifier) Eval(r Vector, x Matrix) error {
  m, _ := x.Dims()
  if r.Dim() != m {
    return fmt.Errorf("r has invalid length")
  }
  if p, err := obj.Viterbi(x); err != nil {
    return err
  } else {
    for i := 0; i < m; i++ {
      r.At(i).SetValue(float64(p[i]))
    }
  }
  return nil
}

func (obj MultiTrackHmmClassifier) Transposed() bool {
  return true
}
