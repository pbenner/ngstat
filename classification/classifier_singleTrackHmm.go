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
import . "github.com/pbenner/ngstat/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type SingleTrackHmmClassifier struct {
  *Hmm
}

/* -------------------------------------------------------------------------- */

func (obj SingleTrackHmmClassifier) CloneSingleTrackClassifier() SingleTrackClassifier {
  return SingleTrackHmmClassifier{obj.Clone()}
}

/* -------------------------------------------------------------------------- */

func (obj SingleTrackHmmClassifier) Dim() int {
  return obj.Hmm.Dim()
}

func (obj SingleTrackHmmClassifier) Eval(r Vector, x Vector) error {
  if r.Dim() != x.Dim() {
    return fmt.Errorf("r has invalid length")
  }
  if p, err := obj.Viterbi(x); err != nil {
    return err
  } else {
    for i := 0; i < x.Dim(); i++ {
      r.At(i).SetValue(float64(p[i]))
    }
  }
  return nil
}
