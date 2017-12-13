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

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type SingleTrackBatchClassifier interface {
  Eval(r Scalar, x Vector) error
  Dim() int
  CloneSingleTrackBatchClassifier() SingleTrackBatchClassifier
}

type MultiTrackBatchClassifier interface {
  Eval(r Scalar, x Matrix) error
  Dims() (int, int)
  CloneMultiTrackBatchClassifier() MultiTrackBatchClassifier
  Transposed() bool
}

/* -------------------------------------------------------------------------- */

type SingleTrackClassifier interface {
  Eval(r Vector, x Vector) error
  Dim() int
  CloneSingleTrackClassifier() SingleTrackClassifier
}

type MultiTrackClassifier interface {
  Eval(r Vector, x Matrix) error
  Dims() (int, int)
  CloneMultiTrackClassifier() MultiTrackClassifier
  Transposed() bool
}
