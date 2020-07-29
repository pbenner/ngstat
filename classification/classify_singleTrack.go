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
import   "os"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/ngstat/track"
import . "github.com/pbenner/ngstat/trackDataTransform"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func ClassifySingleTrackData(config SessionConfig, classifier VectorBatchClassifier, x []Vector, args ...interface{}) ([]float64, error) {
  if len(x) == 0 {
    return nil, nil
  }
  n := classifier.Dim()
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
      return nil, fmt.Errorf("data transform input dimension does not match data dimension")
    } else
    if n2 != classifier.Dim() {
      return nil, fmt.Errorf("data transform output dimension does not match classifier dimension")
    } else {
      m = n2
    }
  } else {
    if n != classifier.Dim() {
      return nil, fmt.Errorf("estimator dimension does not match data dimension")
    }
  }

  pool := threadpool.New(config.Threads, 10000)

  // temporary memory for each thread
  r := NullDenseVector(Float64Type, config.Threads)
  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]VectorBatchClassifier, config.Threads)
  y := make([]Vector, config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneVectorBatchClassifier()
    if f != nil {
      y[i] = NullDenseVector(Float64Type, m)
    }
  }

  g := pool.NewJobGroup()

  result := make([]float64, len(x))

  if err := pool.AddRangeJob(0, len(x), g, func(i int, pool threadpool.ThreadPool, erf func() error) error {
    if erf() != nil {
      return nil
    }
    r := r.At(pool.GetThreadId())
    c := c   [pool.GetThreadId()]
    y := y   [pool.GetThreadId()]
    x := x   [i]
    if n != -1 && x.Dim() != n {
      return fmt.Errorf("dimension of observation `%d' (%d) does not match classifier dimension (%d)", i, x.Dim(), n)
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
    result[i] = r.GetFloat64()
    return nil
  }); err != nil {
    return nil, err
  }
  // wait for threads
  if err := pool.Wait(g); err != nil {
    return nil, err
  }
  return result, nil
}

// Run classifier sequentially on a single track
func BatchClassifySingleTrack(config SessionConfig, classifier VectorBatchClassifier, track Track, args ...interface{}) (MutableTrack, error) {

  var f SingleTrackBatchDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackBatchDataTransform:
      f = a
    }
  }
  if n := classifier.Dim(); n == -1 {
    return nil, fmt.Errorf("classifier must have fixed dimension")
  }
  if n := classifier.Dim(); n < -1 || n == 0 {
    return nil, fmt.Errorf("classifier has invalid dimension")
  }

  n := classifier.Dim()
  m := n

  if f != nil {
    n1, n2 := f.Dims()
    if n2 != classifier.Dim() {
      return nil, fmt.Errorf("data transform output dimension does not match estimator dimension")
    } else {
      n = n1
      m = n2
    }
  }

  nan := math.NaN()

  result := AllocSimpleTrack("classification", track.GetGenome(), track.GetBinSize())
  pool   := threadpool.New(config.Threads, 10000)

  // temporary memory for each thread
  r := NullDenseVector(Float64Type, config.Threads)
  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]VectorBatchClassifier, config.Threads)
  y := make([]Vector, config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneVectorBatchClassifier()
    if f != nil {
      y[i] = NullDenseVector(Float64Type, m)
    }
  }

  offset1 := DivIntUp  (n-1, 2)
  offset2 := DivIntDown(n-1, 2)

  // counter
  l := 0
  // total track length
  L := 0
  for _, length := range track.GetGenome().Lengths {
    L += length/config.BinSize
  }
  if config.Verbose > 0 {
    NewProgress(L, L).PrintStderr(l)
  }

  for _, name := range track.GetSeqNames() {
    seq1, err := track.GetSequence(name); if err != nil {
      return nil, err
    }
    seq2, err := result.GetSequence(name); if err != nil {
      return nil, err
    }
    nbins := seq2.NBins()

    // skip any sequence shorter than the classifier dimension
    if nbins < n {
      for i := 0; i < nbins; i++ {
        seq2.SetBin(i, nan)
      }
      l += nbins

      if config.Verbose > 0 {
        NewProgress(L, L).PrintStderr(l)
      }
      continue
    }
    g := pool.NewJobGroup()

    // convert whole sequence to vector
    x := NullDenseVector(Float64Type, nbins)
    for i := 0; i < nbins; i++ {
      x.At(i).SetFloat64(seq1.AtBin(i))
    }

    // clear non-accessible regions
    for i := 0; i < offset1; i++ {
      seq2.SetBin(i, nan)
    }
    for i := nbins-offset2; i  < nbins; i++ {
      seq2.SetBin(i, nan)
    }
    // launch jobs
    if err := pool.AddRangeJob(offset1, nbins-offset2, g, func(i int, pool threadpool.ThreadPool, erf func() error) error {
      if erf() != nil {
        return nil
      }
      r := r.At(pool.GetThreadId())
      c := c   [pool.GetThreadId()]
      y := y   [pool.GetThreadId()]
      x := x.Slice(i-offset1,i+offset2+1)
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
      seq2.SetBin(i, r.GetFloat64())
      return nil
    }); err != nil {
      return nil, err
    }
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

// Run several independent classifiers and combine results
func BatchClassifySingleTracks(config SessionConfig, classifiers []VectorBatchClassifier, tracks []Track, args ...interface{}) (MutableTrack, error) {

  result, err := BatchClassifySingleTrack(config, classifiers[0], tracks[0], args...); if err != nil {
    return nil, err
  }
  for i := 1; i < len(tracks); i++ {
    tmp, err := BatchClassifySingleTrack(config, classifiers[i], tracks[i], args...); if err != nil {
      return nil, err
    }
    // add tmp track to result
    GenericMutableTrack{result}.MapList([]Track{result, tmp}, func(name string, i int, x... float64) float64 { return x[0]+x[1] })
  }
  return result, nil
}

func ClassifySingleTrack(config SessionConfig, classifier VectorClassifier, track Track, args ...interface{}) (MutableTrack, error) {

  if n := classifier.Dim(); n != -1 {
    return nil, fmt.Errorf("classifier must have variable dimension")
  }
  var f SingleTrackDataTransform

  for _, arg := range args {
    switch a := arg.(type) {
    case SingleTrackDataTransform:
      f = a
    }
  }

  result := AllocSimpleTrack("classification", track.GetGenome(), track.GetBinSize())
  pool   := threadpool.New(config.Threads, 10000)

  // each thread gets its own classifier, since
  // the given classifier may not be thread-safe
  c := make([]VectorClassifier, config.Threads)
  for i := 0; i < config.Threads; i++ {
    c[i] = classifier.CloneVectorClassifier()
  }
  g := pool.NewJobGroup()

  for _, name := range track.GetSeqNames() {
    seq1, err := track.GetSequence(name); if err != nil {
      return nil, err
    }
    seq2, err := result.GetSequence(name); if err != nil {
      return nil, err
    }

    // convert whole sequence to vector
    x := NullDenseVector(Float64Type, seq1.NBins())
    for i := 0; i < seq1.NBins(); i++ {
      x.At(i).SetFloat64(seq1.AtBin(i))
    }
    if f != nil {
      x = f.Eval(x)
    }

    if err := pool.AddJob(g, func(pool threadpool.ThreadPool, erf func() error) error {
      if erf() != nil {
        return nil
      }
      c := c[pool.GetThreadId()]
      r := NullDenseVector(Float64Type, seq1.NBins())
      if err := c.Eval(r, x); err != nil {
        return err
      }
      for i := 0; i < seq1.NBins(); i++ {
        seq2.SetBin(i, r.At(i).GetFloat64())
      }
      return nil
    }); err != nil {
      return nil, err
    }
  }
  // wait for threads
  if err := pool.Wait(g); err != nil {
    return nil, err
  }
  return result, nil
}

/* -------------------------------------------------------------------------- */

func ImportAndBatchClassifySingleTrack(config SessionConfig, classifier VectorBatchClassifier, trackFile string, args ...interface{}) (MutableTrack, error) {
  track, err := ImportTrack(config, trackFile); if err != nil {
    return nil, err
  }
  return BatchClassifySingleTrack(config, classifier, track, args...)
}

func ImportAndBatchClassifySingleTracks(config SessionConfig, classifiers []VectorBatchClassifier, trackFiles []string, args ...interface{}) (MutableTrack, error) {
  var result MutableTrack
  var err    error
  // check in advance if all tracks are available
  for _, file := range trackFiles {
    _, err := os.Stat(file)
    if os.IsNotExist(err) {
      return result, err
    }
    if os.IsPermission(err) {
      return result, err
    }
  }
  // load first track and run classifier
  result, err = ImportAndBatchClassifySingleTrack(config, classifiers[0], trackFiles[0], args...); if err != nil {
    return result, err
  }
  for i := 1; i < len(trackFiles); i++ {
    // load remaining tracks and run classifier
    tmp, err := ImportAndBatchClassifySingleTrack(config, classifiers[i], trackFiles[i], args...); if err != nil {
      return result, err
    }
    // add tmp track to result
    GenericMutableTrack{result}.MapList([]Track{result, tmp}, func(name string, i int, x... float64) float64 { return x[0]+x[1] })
  }
  return result, err
}
