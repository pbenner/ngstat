/* Copyright (C) 2016-2017 Philipp Benner
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

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"
import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/track"
import . "github.com/pbenner/ngstat/trackDataTransform"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func EstimateOnMultiTrackData(config SessionConfig, estimator MatrixEstimator, data []Matrix, transposed bool, args ...interface{}) error {
  var f MultiTrackDataTransform
  var x []Matrix

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackDataTransform:
      f = a
    }
  }

  if transposed || f != nil {
    x = make([]Matrix, len(data))
    for i := 0; i < len(data); i++ {
      x[i] = data[i].CloneMatrix()
    }
  } else {
    x = data
  }
  if transposed {
    for i := 0; i < len(data); i++ {
      x[i].Tip()
    }
  }
  if f != nil {
    for i := 0; i < len(data); i++ {
      x[i] = f.Eval(x[i])
    }
  }

  PrintStderr(config, 1, "Estimating model... ")
  if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
    PrintStderr(config, 1, "failed\n")
    return err
  }
  PrintStderr(config, 1, "done\n")

  return nil
}

func BatchEstimateOnMultiTrackData(config SessionConfig, estimator MatrixBatchEstimator, data []Matrix, transposed bool, args ...interface{}) error {
  var f MultiTrackBatchDataTransform
  var y Matrix

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackBatchDataTransform:
      f = a
    }
  }
  n1, n2 := estimator.Dims()
  m1, m2 := n1, n2

  if f != nil {
    if t1, t2, t3, t4 := f.Dims(); t3 != m1 || t4 != m2 {
      return fmt.Errorf("data transform output dimensions do not match estimator dimension")
    } else {
      n1 = t1
      n2 = t2
    }
  }
  // check data
  for d := 0; d < len(data); d++ {
    if n, _ := data[d].Dims(); n != n1 {
      return fmt.Errorf("data record `%d' has invalid number of rows", d)
    }
    if _, n := data[d].Dims(); n != n2 {
      return fmt.Errorf("data record `%d' has invalid number of columns", d)
    }
  }
  if f != nil {
    y = NullMatrix(estimator.ScalarType(), m1, m2)
  }

  PrintStderr(config, 1, "Estimating model... ")
  for d := 0; d < len(data); d++ {
    x := data[d]
    if transposed {
      x = x.CloneMatrix()
      x.Tip()
    }
    if f != nil {
      if err := f.Eval(y, x); err != nil {
        PrintStderr(config, 1, "failed\n")
        return err
      }
    } else {
      y = x
    }
    if err := estimator.NewObservation(y, nil, ThreadPool{}); err != nil {
      PrintStderr(config, 1, "failed\n")
      return err
    }
  }
  PrintStderr(config, 1, "done\n")

  return nil
}

func BatchEstimateOnMultiTrack(config SessionConfig, estimator MatrixBatchEstimator, tracks []Track, transposed bool, step int, args ...interface{}) error {
  if len(tracks) == 0 {
    return nil
  }
  var f MultiTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackBatchDataTransform:
      f = a
    }
  }
  n1, n2 := estimator.Dims()
  m1, m2 := n1, n2

  if f != nil {
    if t1, t2, t3, t4 := f.Dims(); t3 != m1 || t4 != m2 {
      return fmt.Errorf("data transform output dimensions do not match estimator dimension")
    } else {
      n1 = t1
      n2 = t2
    }
  }
  // set default step size
  if step <= 0 {
    step = n2
  }
  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  y := Matrix(nil)
  if f != nil {
    y = NullMatrix(estimator.ScalarType(), m1, m2)
  }

  offset1 := DivIntUp  (n2-1, 2)
  offset2 := DivIntDown(n2-1, 2)

  // counter
  l := 0
  // total track length
  L := 0
  for _, length := range tracks[0].GetGenome().Lengths {
    L += length/config.BinSize
  }
  if config.Verbose > 0 {
    NewProgress(L, L).PrintStderr(l)
  }
  // memory for collecting track sequences before
  // converting them to vectors
  sequences := make([]TrackSequence, len(tracks))

  for _, name := range tracks[0].GetSeqNames() {
    for k := 0; k < len(tracks); k++ {
      if seq, err := tracks[k].GetSequence(name); err != nil {
        return err
      } else {
        sequences[k] = seq
      }
      if sequences[0].NBins() != sequences[k].NBins() {
        return fmt.Errorf("lengths of sequence `%s' varies between tracks", name)
      }
    }
    x     := SequencesToMatrix(estimator.ScalarType(), sequences, transposed)
    nbins := sequences[0].NBins()

    nrows, ncols := x.Dims()
    // launch jobs
    for i := offset1; i < nbins-offset2; i += step {
      if transposed {
        x := x.Slice(i-offset1, i+offset2+1, 0, ncols)
        if f != nil {
          if err := f.Eval(y, x); err != nil {
            return err
          }
        } else {
          y = x
        }
        if err := estimator.NewObservation(y, nil, ThreadPool{}); err != nil {
          return err
        }
      } else {
        x := x.Slice(0, nrows, i-offset1, i+offset2+1)
        if f != nil {
          if err := f.Eval(y, x); err != nil {
            return err
          }
        } else {
          y = x
        }
        if err := estimator.NewObservation(y, nil, ThreadPool{}); err != nil {
          return err
        }
      }
    }
    l += nbins

    if config.Verbose > 0 {
      NewProgress(L, L).PrintStderr(l)
    }
  }
  return nil
}

func EstimateOnMultiTrack(config SessionConfig, estimator MatrixEstimator, tracks []Track, transposed bool, args ...interface{}) error {
  if len(tracks) == 0 {
    return nil
  }
  if _, m := estimator.Dims(); m != -1 {
    return fmt.Errorf("estimator has wrong dimension (expected variable column dimension, but estimator has dimension `%d')", m)
  }
  var f MultiTrackDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackDataTransform:
      f = a
    }
  }

  x := []Matrix{}
  // collect sequences
LOOP1:
  for _, name := range tracks[0].GetSeqNames() {
    xd := NullVector(BareRealType, 0)
    nd := -1
    for i := 0; i < len(tracks); i++ {
      seq, err := tracks[i].GetSequence(name); if err != nil {
        // skip any sequence that is not present in all tracks
        continue LOOP1
      }
      if nd == -1 {
        nd = seq.NBins()
      }
      if seq.NBins() != nd {
        return fmt.Errorf("sequence `%s' has varying length", name)
      }
      y := NullVector(estimator.ScalarType(), nd)
      for j := 0; j < nd; j++ {
        y.At(j).SetValue(seq.AtBin(j))
      }
      xd = xd.AppendVector(y)
    }
    r := xd.AsMatrix(len(tracks), xd.Dim()/len(tracks))
    if transposed {
      // transpose data in-place to increase performance
      // when accessing rows
      r.Tip()
    }
    if f != nil {
      r = f.Eval(r)
    }
    x = append(x, r)
  }
  PrintStderr(config, 1, "Estimating model... ")
  if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
    PrintStderr(config, 1, "failed\n")
    return err
  } else {
    PrintStderr(config, 1, "done\n")
  }
  return nil
}

/* utility
 * -------------------------------------------------------------------------- */

func ImportAndEstimateOnMultiTrack(config SessionConfig, estimator MatrixEstimator, trackFiles []string, transposed bool, args ...interface{}) error {
  if len(trackFiles) == 0 {
    return nil
  }
  // default parameters
  var args_new []interface{}
  var seqnames map[string]bool

  // parse optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case []string:
      seqnames = make(map[string]bool)
      for _, name := range a {
        seqnames[name] = true
      }
    default:
      args_new = append(args_new, arg)
    }
  }

  tracks := make([]Track, len(trackFiles))

  for i := 0; i < len(trackFiles); i++ {
    if t, err := ImportLazyTrack(config, trackFiles[i]); err != nil {
      return err
    } else {
      if seqnames != nil {
        t.FilterGenome(func(name string, length int) bool {
          return seqnames[name]
        })
      }
      tracks[i] = t; defer t.Close()
    }
  }
  return EstimateOnMultiTrack(config, estimator, tracks, transposed, args_new...)
}

func ImportAndBatchEstimateOnMultiTrack(config SessionConfig, estimator MatrixBatchEstimator, trackFiles []string, transposed bool, step int, args ...interface{}) error {
  if len(trackFiles) == 0 {
    return nil
  }
  // default parameters
  var args_new []interface{}
  var seqnames map[string]bool

  // parse optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case []string:
      seqnames = make(map[string]bool)
      for _, name := range a {
        seqnames[name] = true
      }
    default:
      args_new = append(args_new, arg)
    }
  }

  tracks := make([]Track, len(trackFiles))

  for i := 0; i < len(trackFiles); i++ {
    if t, err := ImportLazyTrack(config, trackFiles[i]); err != nil {
      return err
    } else {
      if seqnames != nil {
        t.FilterGenome(func(name string, length int) bool {
          return seqnames[name]
        })
      }
      tracks[i] = t; defer t.Close()
    }
  }
  return BatchEstimateOnMultiTrack(config, estimator, tracks, transposed, step, args_new...)
}