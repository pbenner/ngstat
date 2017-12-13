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

package track

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "io"
import   "sort"

import . "github.com/pbenner/ngstat/config"

import . "github.com/pbenner/gonetics"
//import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func segmentationHistogram(config SessionConfig, segmentation Track, tracks []Track, nstates int) ([][]map[float64]int, error) {
  values := make([][]map[float64]int, len(tracks))
  for i, _ := range tracks {
    values[i] = make([]map[float64]int, nstates)
    for j := 0; j < nstates; j++ {
      values[i][j] = make(map[float64]int)
    }
  }
  err := (GenericMutableTrack{}).MapList(append([]Track{segmentation}, tracks...),
    func(seqname string, position int, v ...float64) float64 {
      state := int(v[0])
      for i := 1; i < len(v); i++ {
        values[i-1][state][v[i]] += 1
      }
      return 0.0
    })
  if err != nil {
    return nil, err
  } else {
    return values, nil
  }
}

func SegmentationHistogram(config SessionConfig, segmentationFilename string, trackFilenames []string, nstates int, genome Genome) ([][]map[float64]int, error) {

  tracks := []Track{}

  for _, filename := range trackFilenames {

    if t, err := ImportLazyTrack(config, filename); err != nil {
      panic(err)
    } else {
      tracks = append(tracks, t)
    }
  }
  if segmentation, err := ImportTrackSegmentation(config, segmentationFilename, genome); err != nil {
    panic(err)
  } else {
    return segmentationHistogram(config, segmentation, tracks, nstates)
  }
}

/* -------------------------------------------------------------------------- */

func writeSegmentationData(w io.Writer, data map[float64]int, i, j int) {
  keys := []float64{}
  for v, _ := range data {
    keys = append(keys, v)
  }
  sort.Float64s(keys)
  for _, v := range keys {
    fmt.Fprintf(w, "%10v %10v %20v %15v\n", i, j, v, data[v])
  }
}

func WriteSegmentationHistogram(config SessionConfig, w io.Writer, segmentationFilename string, trackFilenames []string, nstates int, genome Genome) error {
  if values, err := SegmentationHistogram(config, segmentationFilename, trackFilenames, nstates, genome); err != nil {
    return err
  } else {
    // loop over data
    fmt.Fprintf(w, "%10s %10s %20s %15s\n", "track", "state", "value", "count")
    for i := 0; i < len(values); i++ {
      for j := 0; j < nstates; j++ {
        writeSegmentationData(w, values[i][j], i, j)
      }
    }
    return nil
  }
}
