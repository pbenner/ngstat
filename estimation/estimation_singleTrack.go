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

package estimation

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"
import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/trackDataTransform"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func EstimateOnSingleTrackData(config SessionConfig, estimator VectorEstimator, data []Vector, args ...interface{}) error {
  if len(data) == 0 {
    return nil
  }

  var f SingleTrackDataTransform
  var x []Vector

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackDataTransform:
      f = a
    }
  }

  if f != nil {
    x = make([]Vector, len(data))
    for i := 0; i < len(data); i++ {
      x[i] = f.Eval(data[i].CloneVector())
    }
  } else {
    x = data
  }

  PrintStderr(config, 1, "Estimating model... ")
  if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
    PrintStderr(config, 1, "failed\n")
    return err
  }
  PrintStderr(config, 1, "done\n")

  return nil
}

func BatchEstimateOnSingleTrackData(config SessionConfig, estimator VectorBatchEstimator, data []Vector, args ...interface{}) error {
  if len(data) == 0 {
    return nil
  }
  n := estimator.Dim()
  m := n

  var f SingleTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackBatchDataTransform:
      f = a
    }
  }
  // check dimensions
  if f != nil {
    if n1, n2 := f.Dims(); n1 != n {
      return fmt.Errorf("data transform input dimension does not match data dimension")
    } else
    if n2 != estimator.Dim() {
      return fmt.Errorf("data transform output dimension does not match estimator dimension")
    } else {
      m = n2
    }
  } else {
    if n != estimator.Dim() {
      return fmt.Errorf("estimator dimension does not match data dimension")
    }
  }
  // temporary memory
  var y Vector
  if f != nil {
    y = NullVector(estimator.ScalarType(), m)
  }
  if err := estimator.Initialize(ThreadPool{}); err != nil {
    return err
  }

  PrintStderr(config, 1, "Estimating model... ")
  for d := 0; d < len(data); d++ {
    x := data[d]

    if x.Dim() != n {
      PrintStderr(config, 1, "failed\n")
      return fmt.Errorf("dimension of observation %d does not match estimator dimension", d)
    }
    if f != nil {
      if err := f.Eval(y, x); err != nil {
        PrintStderr(config, 1, "failed\n")
        return err
      }
      if err := estimator.NewObservation(y, nil, ThreadPool{}); err != nil {
        PrintStderr(config, 1, "failed\n")
        return err
      }
    } else {
      if err := estimator.NewObservation(x, nil, ThreadPool{}); err != nil {
        PrintStderr(config, 1, "failed\n")
        return err
      }
    }
  }
  PrintStderr(config, 1, "done\n")

  return nil
}

/* -------------------------------------------------------------------------- */

func BatchEstimateOnSingleTrack(config SessionConfig, estimator VectorBatchEstimator, track Track, step int, args ...interface{}) error {
  var f SingleTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackBatchDataTransform:
      f = a
    }
  }
  n := estimator.Dim()
  m := n

  if f == nil {
    if n != config.WindowSize/config.BinSize {
      return fmt.Errorf("estimator has wrong dimension (expected dimension `%d', but estimator has dimension `%d'", m, estimator.Dim())
    }
  } else {
    n1, n2 := f.Dims()
    if n1 != config.WindowSize/config.BinSize {
      return fmt.Errorf("data transform has wrong input dimension (expected dimension `%d', but transform has dimension `%d'", config.WindowSize/config.BinSize, n1)
    }
    if n2 != m {
      return fmt.Errorf("data transform output dimension (%d) does not match estimator dimension (%d)", n2, m)
    }
    n = n1
  }
  // set default step size
  if step <= 0 {
    step = n
  }
  // allocate matrix where track slices are copied to
  x := NullVector(estimator.ScalarType(), n)
  // storage for transformed data
  y := x
  if f != nil {
    y = NullVector(estimator.ScalarType(), m)
  }
  if err := estimator.Initialize(ThreadPool{}); err != nil {
    return err
  }
  // counter
  l := 0
  // total track length
  L := 0
  for _, length := range track.GetGenome().Lengths {
    L += length/config.BinSize
  }
  PrintStderr(config, 1, "Estimating model... \n")
  if config.Verbose >= 1 {
    NewProgress(L, L).PrintStderr(l)
  }
  for _, name := range track.GetSeqNames() {
    seq, err := track.GetSequence(name); if err != nil {
      return err
    }
    // launch jobs
  LOOP_SEQUENCE:
    for i := 0; i < seq.NBins()-n; i += step {
      // copy track slice to x and check for
      // masked regions
      for j := 0; j < n; j++ {
        if math.IsNaN(seq.AtBin(i+j)) {
          continue LOOP_SEQUENCE
        }
        x.At(j).SetValue(seq.AtBin(i+j))
      }
      if f != nil {
        if err := f.Eval(y, x); err != nil {
          return err
        }
      }
      estimator.NewObservation(y, nil, ThreadPool{})
    }
    l += seq.NBins()

    if config.Verbose >= 1 {
      NewProgress(L, L).PrintStderr(l)
    }
  }

  return nil
}

func EstimateOnSingleTrack(config SessionConfig, estimator VectorEstimator, track Track, step int, args ...interface{}) error {
  if estimator.Dim() != -1 {
    return fmt.Errorf("estimator has wrong dimension (expected variable dimension, but estimator has dimension `%d'", estimator.Dim())
  }
  var f SingleTrackDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackDataTransform:
      f = a
    }
  }

  x := []Vector{}
  // collect sequences
  for _, name := range track.GetSeqNames() {
    seq, err := track.GetSequence(name); if err != nil {
      return err
    }
    y := NullVector(estimator.ScalarType(), seq.NBins())
    for i := 0; i < seq.NBins(); i++ {
      y.At(i).SetValue(seq.AtBin(i))
    }
    if f != nil {
      y = f.Eval(y)
    }
    x = append(x, y)
  }
  PrintStderr(config, 1, "Estimating model... ")
  estimator.EstimateOnData(x, nil, ThreadPool{})
  PrintStderr(config, 1, "done\n")

  return nil
}
