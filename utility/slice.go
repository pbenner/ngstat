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

package utility

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func Flatten(a [][]float64) []float64 {
  r := []float64{}
  for _, v := range a {
    r = append(r, v...)
  }
  return r
}

func ShiftRing(s []float64, i int) []float64 {
  if i >= 0 {
    return append(s[len(s)-i:len(s)], s[0:len(s)-i]...)
  } else {
    return append(s[-i:len(s)], s[0:-i]...)
  }
}

func Argmax(s []float64) int {
  j := 0
  for i := 1; i < len(s); i++ {
    if s[i] > s[j] {
      j = i
    }
  }
  return j
}

func SliceMax(s []float64) float64 {
  j := 0
  for i := 1; i < len(s); i++ {
    if s[i] > s[j] {
      j = i
    }
  }
  return s[j]
}

func SliceMin(s []float64) float64 {
  j := 0
  for i := 1; i < len(s); i++ {
    if s[i] < s[j] {
      j = i
    }
  }
  return s[j]
}

func SliceMean(s []float64) float64 {
  sum := 0.0
  for i := 0; i < len(s); i++ {
    sum += s[i]
  }
  return sum/float64(len(s))
}

/* -------------------------------------------------------------------------- */

func SlicesToVectors(t ScalarType, counts [][]float64, transposed bool) []Vector {
  var n int
  var x []Vector
  // exit if no data is available
  if len(counts) == 0 {
    return x
  }
  if transposed {
    n = len(counts)
    x = []Vector{}
    for j := 0; j < len(counts[0]); j++ {
      v := NullDenseVector(t, n)
      for i := 0; i < len(counts); i++ {
        v.At(i).SetFloat64(counts[i][j])
      }
      x = append(x, v)
    }
  } else {
    n = len(counts)
    // extract 2-dimensional data
    x = make([]Vector, n)
    for i := 0; i < n; i++ {
      v := NullDenseVector(t, len(counts[i]))
      for j := 0; j < len(counts[i]); j++ {
        v.At(j).SetFloat64(counts[i][j])
      }
      x[i] = v
    }
  }
  return x
}

func SequencesToVectors(t ScalarType, sequences []TrackSequence, transposed bool) []Vector {
  var n int
  var x []Vector
  // exit if no data is available
  if len(sequences) == 0 {
    return x
  }
  if transposed {
    n = len(sequences)
    x = []Vector{}
    for j := 0; j < sequences[0].NBins(); j++ {
      v := NullDenseVector(t, n)
      for i := 0; i < len(sequences); i++ {
        v.At(i).SetFloat64(sequences[i].AtBin(j))
      }
      x = append(x, v)
    }
  } else {
    n = len(sequences)
    // extract 2-dimensional data
    x = make([]Vector, n)
    for i := 0; i < n; i++ {
      x[i] = NullDenseVector(t, sequences[i].NBins())
      for j := 0; j < sequences[i].NBins(); j++ {
        x[i].At(j).SetFloat64(sequences[i].AtBin(j))
      }
    }
  }
  return x
}

func SequencesToMatrix(t ScalarType, sequences []TrackSequence, transposed bool) Matrix {
  if len(sequences) == 0 {
    return NullDenseMatrix(t, 0, 0)
  }
  if transposed {
    x := NullDenseMatrix(t, sequences[0].NBins(), len(sequences))
    for i := 0; i < len(sequences); i++ {
      for j := 0; j < sequences[i].NBins(); j++ {
        x.At(j,i).SetFloat64(sequences[i].AtBin(j))
      }
    }
    return x
  } else {
    x := NullDenseMatrix(t, len(sequences), sequences[0].NBins())
    for i := 0; i < len(sequences); i++ {
      for j := 0; j < sequences[i].NBins(); j++ {
        x.At(i,j).SetFloat64(sequences[i].AtBin(j))
      }
    }
    return x
  }
}

/* -------------------------------------------------------------------------- */

func SlicesToMatrix(t ScalarType, counts [][]float64, transposed bool) Matrix {
  x := NullDenseVector(t, 0)
  n := len(counts)
  // exit if no data is available
  if n == 0 {
    return x.AsMatrix(0, 0)
  }
  if transposed {
    for j := 0; j < len(counts[0]); j++ {
      v := NullDenseVector(t, n)
      for i := 0; i < len(counts); i++ {
        v.At(i).SetFloat64(counts[i][j])
      }
      x = x.AppendVector(v)
    }
    return x.AsMatrix(x.Dim()/n, n)
  } else {
    n = len(counts)
    // extract 2-dimensional data
    x = NullDenseVector(t, 0)
    for i := 0; i < n; i++ {
      v := NullDenseVector(t, len(counts[i]))
      for j := 0; j < len(counts[i]); j++ {
        v.At(j).SetFloat64(counts[i][j])
      }
      x = x.AppendVector(v)
    }
    return x.AsMatrix(n, x.Dim()/n)
  }
}
