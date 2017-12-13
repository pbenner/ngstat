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

import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func sortPeaks(peaks GRanges) GRanges {
  // sum up test results for sorting rows
  peaks.ReduceFloat("test","test.sum", func(x []float64) float64 {
    sum := 0.0
    for i := 0; i < len(x); i++ {
      sum += x[i]
    }
    return sum
  })
  peaks, _ = peaks.Sort("test.sum", true)
  peaks.DeleteMeta("test.sum")

  return peaks
}

func intersectingPeaks(r, s GRanges) GRanges {
  queryHits, subjectHits := FindOverlaps(r, s)
  n        := len(queryHits)
  seqnames := make([]string, n)
  from     := make([]int, n)
  to       := make([]int, n)
  strand   := make([]byte, n)
  test     := make([][]float64, n)
  // get test results
  rTest := r.GetMeta("test").([][]float64)
  sTest := s.GetMeta("test").(  []float64)

  for i := 0; i < n; i++ {
    iQ := queryHits[i]
    iS := subjectHits[i]
    gr := r.Ranges[iQ].Intersection(s.Ranges[iS])
    seqnames[i] = r.Seqnames[iQ]
    strand  [i] = r.Strand  [iQ]
    from    [i] = gr.From
    to      [i] = gr.To
    test    [i] = append(rTest[iQ], sTest[iS])
  }
  result := NewGRanges(seqnames, from, to, strand)
  result.AddMeta("test", test)

  return result
}

func GetPeaks(track Track, threshold float64, wsize int) GRanges {
  seqnames := []string{}
  from     := []int{}
  to       := []int{}
  strand   := []byte{}
  test     := []float64{}

  offset1 := DivIntUp  (wsize-1, 2)
  offset2 := DivIntDown(wsize-1, 2)

  for _, name := range track.GetSeqNames() {
    sequence, err := track.GetSequence(name); if err != nil {
      continue
    }
    for i := 0; i < sequence.NBins(); i++ {
      if sequence.AtBin(i) > threshold {
        // peak begins here
        i_from := i
        // maximum value
        v_max  := sequence.AtBin(i)
        // position of the maximum value
        i_max  := i
        // increment until either the sequence ended or
        // the value drops below the threshold
        for i < sequence.NBins() && sequence.AtBin(i) > threshold {
          if sequence.AtBin(i) > v_max {
            // update maximum position and value
            i_max = i
            v_max = sequence.AtBin(i)
          }
          i += 1
        }
        seqnames = append(seqnames, name)
        test     = append(test, sequence.AtBin(i_max))
        if wsize > 0 {
          // cut a window around the maximum
          tFrom := i_max*track.GetBinSize()-offset1
          tTo   := i_max*track.GetBinSize()+offset2+1
          if tFrom < 0 {
            tFrom = 0
          }
          if tTo > sequence.NBins()*track.GetBinSize() {
            tTo = sequence.NBins()*track.GetBinSize()
          }
          from = append(from, tFrom)
          to   = append(to,   tTo)
        } else {
          // save full peak
          tFrom := i_from*track.GetBinSize()-offset1
          tTo   := i     *track.GetBinSize()+offset2+1
          from = append(from, tFrom)
          to   = append(to,   tTo)
        }
      }
    }
  }
  peaks := NewGRanges(seqnames, from, to, strand)
  peaks.AddMeta("test", test)
  peaks, _ = peaks.Sort("test", true)

  return peaks
}

func GetJointPeaks(tracks []Track, thresholds []float64, wsize int) (GRanges, error) {
  if len(tracks) != len(thresholds) {
    return GRanges{}, fmt.Errorf("GetJointPeaks(): invalid arguments")
  }
  if len(tracks) == 0 {
    return GRanges{}, nil
  }
  seqnames := []string{}
  from     := []int{}
  to       := []int{}
  strand   := []byte{}
  test     := [][]float64{}

  offset1 := DivIntUp  (wsize-1, 2)
  offset2 := DivIntDown(wsize-1, 2)

  binsize := tracks[0].GetBinSize()
  // assert that all tracks have the same binsize
  for j := 1; j < len(tracks); j++ {
    if binsize != tracks[j].GetBinSize() {
      return GRanges{}, fmt.Errorf("tracks `1' and `%d' have different bin sizes (`%d' and `%d')", j+1, binsize, tracks[j].GetBinSize())
    }
  }
  for _, name := range tracks[0].GetSeqNames() {
    s, err := tracks[0].GetSequence(name); if err != nil {
      return GRanges{}, err
    }
    sequences := []TrackSequence{s}
    seqlen    := s.NBins()
    // check remaining track for consistency
    for j := 1; j < len(tracks); j++ {
      if binsize != tracks[j].GetBinSize() {
        return GRanges{}, fmt.Errorf("tracks `1' and `%d' have different bin sizes (`%d' and `%d')", j+1, binsize, tracks[j].GetBinSize())
      }
      if sequence, err := tracks[j].GetSequence(name); err != nil {
        return GRanges{}, fmt.Errorf("reading sequence from track `%d' failed: %v", j+1, err)
      } else {
        if sequence.NBins() != seqlen {
          return GRanges{}, fmt.Errorf("sequence `%s' on track `1' and `%d' have different lengths (`%d' and `%d')", name, j+1, seqlen, sequence.NBins())
        }
        sequences = append(sequences, sequence)
      }
    }
    for i := 0; i < seqlen; i++ {
      if allPositive(sequences, thresholds, i) {
        // peak begins here
        i_from := i
        // maximum value
        v_max  := allSum(sequences, i)
        // position of the maximum value
        i_max  := i
        // increment until either the sequence ended or
        // the value drops below the threshold
        for i < seqlen && allPositive(sequences, thresholds, i) {
          if sum := allSum(sequences, i); sum > v_max {
            // update maximum position and value
            i_max = i
            v_max = sum
          }
          i += 1
        }
        tmp := make([]float64, len(sequences))
        for j := 0; j < len(sequences); j++ {
          tmp[j] = sequences[j].AtBin(i_max)
        }
        test     = append(test, tmp)
        seqnames = append(seqnames, name)
        if wsize > 0 {
          // cut a window around the maximum
          tFrom := i_max*binsize-offset1
          tTo   := i_max*binsize+offset2+1
          if tFrom < 0 {
            tFrom = 0
          }
          if tTo > seqlen*binsize {
            tTo = seqlen*binsize
          }
          from = append(from, tFrom)
          to   = append(to,   tTo)
        } else {
          // save full peak
          tFrom := i_from*binsize-offset1
          tTo   := i     *binsize+offset2+1
          from = append(from, tFrom)
          to   = append(to,   tTo)
        }
      }
    }
  }
  peaks := NewGRanges(seqnames, from, to, strand)
  peaks.AddMeta("test", test)
  // sum up test results for sorting rows
  peaks.ReduceFloat("test","test.sum", func(x []float64) float64 {
    sum := 0.0
    for i := 0; i < len(x); i++ {
      sum += x[i]
    }
    return sum
  })
  peaks, _ = peaks.Sort("test.sum", true)
  peaks.DeleteMeta("test.sum")

  return peaks, nil
}

func GetIntersectingPeaks(tracks []Track, thresholds []float64, wsize int) (GRanges, error) {
  var peaks GRanges

  if len(tracks) != len(thresholds) {
    return peaks, fmt.Errorf("GetPredictions(): invalid arguments")
  }
  if len(tracks) == 0 {
    return peaks, nil
  }
  peaks = GetPeaks(tracks[0], thresholds[0], wsize)
  // convert test from []float64 to [][]float64
  {
    aTest := peaks.GetMeta("test").([]float64)
    bTest := make([][]float64, len(aTest))
    for i := 0; i < len(aTest); i++ {
      bTest[i] = []float64{aTest[i]}
    }
    peaks.DeleteMeta("test")
    peaks.AddMeta("test", bTest)
  }
  for i := 1; i < len(tracks); i++ {
    tmp  := GetPeaks(tracks[i], thresholds[i], wsize)
    peaks = intersectingPeaks(peaks, tmp)
  }
  // if window size is given, resize all peaks to wsize
  if wsize > 0 {
    offset1 := DivIntUp  (wsize-1, 2)
    offset2 := DivIntDown(wsize-1, 2)

    for i := 0; i < peaks.Length(); i++ {
      center := (peaks.Ranges[i].From + peaks.Ranges[i].To - 1)/2
      peaks.Ranges[i].From = center-offset1
      peaks.Ranges[i].To   = center+offset2+1
    }
  }
  return sortPeaks(peaks), nil
}

/* old implementation
 * -------------------------------------------------------------------------- */

func filterOverlaps(granges GRanges) GRanges {

  queryHits, subjectHits := FindOverlaps(granges, granges)

  test := granges.GetMetaFloat("test")
  idx  := []int{}

  for i := 0; i < len(queryHits); i++ {
    j1 :=   queryHits[i]
    j2 := subjectHits[i]
    if j1 != j2 {
      if test[j1] > test[j2] {
        idx = append(idx, j2)
      } else {
      idx = append(idx, j1)
      }
    }
  }
  return granges.Remove(idx)
}

func allPositive(sequences []TrackSequence, thresholds []float64, i int) bool {
  for j := 0; j < len(sequences); j++ {
    if math.IsNaN(sequences[j].AtBin(i)) || sequences[j].AtBin(i) <= thresholds[j] {
      return false
    }
  }
  return true
}

func allSum(sequences []TrackSequence, i int) float64 {
  sum := 0.0
  for j := 0; j < len(sequences); j++ {
    sum += sequences[j].AtBin(i)
  }
  return sum
}

func GetPredictions(tracks []Track, thresholds []float64, wsize int) (GRanges, error) {

  if len(tracks) != len(thresholds) {
    return GRanges{}, fmt.Errorf("GetPredictions(): invalid arguments")
  }
  if len(tracks) == 0 {
    return GRanges{}, nil
  }
  seqnames := []string{}
  from     := []int{}
  to       := []int{}
  strand   := []byte{}
  test     := [][]float64{}

  offset1 := DivIntUp  (wsize-1, 2)
  offset2 := DivIntDown(wsize-1, 2)

  for _, name := range tracks[0].GetSeqNames() {
    s, err := tracks[0].GetSequence(name); if err != nil {
      return GRanges{}, err
    }
    sequences := []TrackSequence{}
    seqlen    := s.NBins()
    for j := 0; j < len(tracks); j++ {
      if tracks[0].GetBinSize() != tracks[j].GetBinSize() {
        return GRanges{}, fmt.Errorf("tracks `1' and `%d' have different bin sizes (`%d' and `%d')", j+1, tracks[0].GetBinSize(), tracks[j].GetBinSize())
      }
      if sequence, err := tracks[j].GetSequence(name); err != nil {
        return GRanges{}, fmt.Errorf("reading sequence from track `%d' failed: %v", j+1, err)
      } else {
        if sequence.NBins() != seqlen {
          return GRanges{}, fmt.Errorf("sequence `%s' on track `1' and `%d' have different lengths (`%d' and `%d')", name, j+1, seqlen, sequence.NBins())
        }
        sequences = append(sequences, sequence)
      }
    }
    for i := 0; i < seqlen; i++ {
      if allPositive(sequences, thresholds, i) {
        tFrom := i*tracks[0].GetBinSize()-offset1
        tTo   := i*tracks[0].GetBinSize()+offset2+1
        if tFrom < 0 {
          tFrom = 0
        }
        if tTo > seqlen*tracks[0].GetBinSize() {
          tTo = seqlen*tracks[0].GetBinSize()
        }
        seqnames = append(seqnames, name)
        from     = append(from, tFrom)
        to       = append(to,   tTo)
        test     = append(test, []float64{})
        for j, n := 0, len(test)-1; j < len(sequences); j++ {
          test[n] = append(test[n], sequences[j].AtBin(i))
        }
      }
    }
  }
  granges := NewGRanges(seqnames, from, to, strand)
  granges.AddMeta("test", test)
  granges  = filterOverlaps(granges)
  // sum up test results for sorting rows
  granges.ReduceFloat("test","test.sum", func(x []float64) float64 {
    sum := 0.0
    for i := 0; i < len(x); i++ {
      sum += x[i]
    }
    return sum
  })
  granges, _ = granges.Sort("test.sum", true)
  granges.DeleteMeta("test.sum")

  return granges, nil
}
