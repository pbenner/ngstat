/* Copyright (C) 2016, 2017 Philipp Benner
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

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/track"
import . "github.com/pbenner/ngstat/trackDataTransform"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func ClassifyMultiTrackData(config SessionConfig, classifier MatrixBatchClassifier, data []Matrix, transposed bool, args ...interface{}) ([]float64, error) {

  var f MultiTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackBatchDataTransform:
      f = a
    }
  }
  n1, n2 := classifier.Dims()
  m1, m2 := n1, n2

  if f != nil {
    if t1, t2, t3, t4 := f.Dims(); t3 != m1 || t4 != m2 {
      return nil, fmt.Errorf("data transform output dimensions do not match classifier dimension")
    } else {
      n1 = t1
      n2 = t2
    }
  }
  // check data
  for d := 0; d < len(data); d++ {
    if n, _ := data[d].Dims(); n != n1 {
      return nil, fmt.Errorf("data record `%d' has invalid number of rows", d)
    }
    if _, n := data[d].Dims(); n != n2 {
      return nil, fmt.Errorf("data record `%d' has invalid number of columns", d)
    }
  }

  // temporary memory for each thread
  r := NullVector(BareRealType, config.Threads)
  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]MatrixBatchClassifier, config.Threads)
  y := make([]Matrix,                    config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneMatrixBatchClassifier()
    if f != nil {
      y[i] = NullMatrix(BareRealType, m1, m2)
    }
  }
  result := make([]float64, len(data))

  pool := NewThreadPool(config.Threads, 10000)
  g    := pool.NewJobGroup()
  // classify data
  for d := 0; d < len(data); d++ {
    // thread safe copy of d
    d := d
    pool.AddJob(g, func(pool ThreadPool, erf func() error) error {
      c := c   [pool.GetThreadId()]
      y := y   [pool.GetThreadId()]
      r := r.At(pool.GetThreadId())
      if erf() != nil {
        return nil
      }
      x := data[d]
      if transposed {
        x.Tip()
      }
      if f != nil {
        if err := f.Eval(y, x); err != nil {
          return err
        }
      } else {
        y = x
      }
      if err := c.Eval(r, y); err != nil {
        return err
      }
      result[d] = r.GetValue()
      return nil
    })
  }
  // wait for threads
  if err := pool.Wait(g); err != nil {
    return nil, err
  }
  return result, nil
}

func BatchClassifyMultiTrack(config SessionConfig, classifier MatrixBatchClassifier, tracks []Track, transposed bool, args ...interface{}) (MutableTrack, error) {

  if len(tracks) == 0 {
    return nil, nil
  }
  var f MultiTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackBatchDataTransform:
      f = a
    }
  }
  n1, n2 := classifier.Dims()
  m1, m2 := n1, n2

  if f != nil {
    if t1, t2, t3, t4 := f.Dims(); t3 != m1 || t4 != m2 {
      return nil, fmt.Errorf("data transform output dimensions do not match classifier dimension")
    } else {
      n1 = t1
      n2 = t2
    }
  }
  if len(tracks) != n1 {
    return nil, fmt.Errorf("invalid number of tracks (expected `%d' tracks, but `%d' are given)", n1, len(tracks))
  }

  nan := math.NaN()

  result := AllocSimpleTrack("classification", tracks[0].GetGenome(), tracks[0].GetBinSize())
  pool   := NewThreadPool(config.Threads, 10000)

  // temporary memory for each thread
  r := NullVector(BareRealType, config.Threads)
  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]MatrixBatchClassifier, config.Threads)
  y := make([]Matrix,                    config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneMatrixBatchClassifier()
    if f != nil {
      y[i] = NullMatrix(BareRealType, m1, m2)
    }
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
    dst, err := result.GetSequence(name); if err != nil {
      return nil, err
    }
    nbins := dst.NBins()

    // skip any sequence shorter than the classifier dimension
    if nbins < n2 {
      for i := 0; i < nbins; i++ {
        dst.SetBin(i, nan)
      }
      l += nbins

      if config.Verbose > 0 {
        NewProgress(L, L).PrintStderr(l)
      }
      continue
    }

    for k := 0; k < len(tracks); k++ {
      seq, err := tracks[k].GetSequence(name); if err != nil {
        return nil, err
      }
      if nbins != seq.NBins() {
        return nil, fmt.Errorf("lengths of sequence `%s' varies between tracks", name)
      }
      sequences[k] = seq
    }
    g := pool.NewJobGroup()
    x := SequencesToMatrix(BareRealType, sequences, transposed)

    // clear non-accessible regions
    for i := 0; i < offset1; i++ {
      dst.SetBin(i, nan)
    }
    for i := nbins-offset2; i < nbins; i++ {
      dst.SetBin(i, nan)
    }
    nrows, ncols := x.Dims()
    // launch jobs
    pool.AddRangeJob(offset1, nbins-offset2, g, func(i int, pool ThreadPool, erf func() error) error {
      if erf() != nil {
        return nil
      }
      c := c   [pool.GetThreadId()]
      y := y   [pool.GetThreadId()]
      r := r.At(pool.GetThreadId())
      if transposed {
        x := x.Slice(i-offset1, i+offset2+1, 0, ncols)
        if f != nil {
          if err := f.Eval(y, x); err != nil {
            return err
          }
        } else {
          y = x
        }
        if err := c.Eval(r, y); err != nil {
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
        if err := c.Eval(r, y); err != nil {
          return err
        }
      }
      dst.SetBin(i, r.GetValue())
      return nil
    })
    // wait for threads
    if err := pool.Wait(g); err != nil {
      return nil, err
    }
    l += nbins

    if config.Verbose > 0 {
      NewProgress(L, L).PrintStderr(l)
    }
  }
  return result, nil
}

func ClassifyMultiTrack(config SessionConfig, classifier MatrixClassifier, tracks []Track, transposed bool, args ...interface{}) (MutableTrack, error) {

  if _, n := classifier.Dims(); n != -1 {
    return nil, fmt.Errorf("classifier must have variable column dimension")
  }
  if len(tracks) == 0 {
    return nil, nil
  }
  var f MultiTrackDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case MultiTrackDataTransform:
      f = a
    }
  }

  result := AllocSimpleTrack("classification", tracks[0].GetGenome(), tracks[0].GetBinSize())
  pool   := NewThreadPool(config.Threads, 10000)

  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]MatrixClassifier, config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneMatrixClassifier()
  }

  // memory for collecting track sequences before
  // converting them to vectors
  sequences := make([]TrackSequence, len(tracks))

  g := pool.NewJobGroup()

  for _, name := range tracks[0].GetSeqNames() {
    dst, err := result.GetSequence(name); if err != nil {
      return nil, err
    }
    nbins := dst.NBins()

    for k := 0; k < len(tracks); k++ {
      seq, err := tracks[k].GetSequence(name); if err != nil {
        return nil, err
      }
      if nbins != seq.NBins() {
        return nil, fmt.Errorf("lengths of sequence `%s' varies between tracks", name)
      }
      sequences[k] = seq
    }
    r := NullVector(BareRealType, nbins)
    x := SequencesToMatrix(BareRealType, sequences, transposed)
    if f != nil {
      x = f.Eval(x)
    }

    pool.AddJob(g, func(pool ThreadPool, erf func() error) error {
      c := c[pool.GetThreadId()]
      if erf() != nil {
        return nil
      }
      if err := c.Eval(r, x); err != nil {
        return err
      }
      for i := 0; i < nbins; i++ {
        dst.SetBin(i, r.At(i).GetValue())
      }
      return nil
    })
  }
  // wait for threads
  if err := pool.Wait(g); err != nil {
    return nil, err
  }
  return result, nil
}

/* -------------------------------------------------------------------------- */

func ImportAndBatchClassifyMultiTrack(config SessionConfig, classifier MatrixBatchClassifier, trackFiles []string, transposed bool, args ...interface{}) (MutableTrack, error) {
  tracks := make([]Track, len(trackFiles))
  for i := 0; i < len(trackFiles); i++ {
    track, err := ImportLazyTrack(config, trackFiles[i]); if err != nil {
      return nil, err
    }
    tracks[i] = track
  }
  return BatchClassifyMultiTrack(config, classifier, tracks, transposed, args...)
}

func ImportAndClassifyMultiTrack(config SessionConfig, classifier MatrixClassifier, trackFiles []string, transposed bool, args ...interface{}) (MutableTrack, error) {
  tracks := make([]Track, len(trackFiles))
  for i := 0; i < len(trackFiles); i++ {
    track, err := ImportLazyTrack(config, trackFiles[i]); if err != nil {
      return nil, err
    }
    tracks[i] = track
  }
  return ClassifyMultiTrack(config, classifier, tracks, transposed, args...)
}
