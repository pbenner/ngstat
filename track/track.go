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

package track

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "os"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func ImportTrack(config SessionConfig, trackFilename string) (SimpleTrack, error) {
  track := SimpleTrack{}
  PrintStderr(config, 1, "Reading track `%s'... ", trackFilename)
  if l, err := config.GetBinSummaryStatistics(); err != nil {
    return track, err
  } else {
    if err := track.ImportBigWig(trackFilename, "", l, config.BinSize, config.BinOverlap, config.TrackInit); err != nil {
      PrintStderr(config, 1, "failed\n")
      return track, err
    }
  }
  PrintStderr(config, 1, "done\n")
  return track, nil
}

func ImportLazyTrack(config SessionConfig, trackFilename string) (LazyTrackFile, error) {
  track := LazyTrackFile{}
  PrintStderr(config, 1, "Lazy importing track `%s'... ", trackFilename)
  if l, err := config.GetBinSummaryStatistics(); err != nil {
    return track, err
  } else {
    if err := track.ImportBigWig(trackFilename, "", l, config.BinSize, config.BinOverlap, config.TrackInit); err != nil {
      PrintStderr(config, 1, "failed\n")
      return track, err
    }
  }
  PrintStderr(config, 1, "done\n")
  return track, nil
}

func ImportTrackRegions(config SessionConfig, trackFilename, bedFilename string) (GRanges, error) {
  r := GRanges{}
  PrintStderr(config, 1, "Reading bed file `%s'... ", bedFilename)
  if err := r.ImportBed3(bedFilename); err != nil {
    PrintStderr(config, 1, "failed\n")
    return r, err
  } else {
    PrintStderr(config, 1, "done\n")
  }

  PrintStderr(config, 1, "Importing regions from track `%s'... ", trackFilename)
  if l, err := config.GetBinSummaryStatistics(); err != nil {
    PrintStderr(config, 1, "failed\n")
    return r, err
  } else {
    if err := r.ImportBigWig(trackFilename, "counts", l, config.BinSize, config.BinOverlap, config.TrackInit, false); err != nil {
      PrintStderr(config, 1, "failed\n")
      return r, err
    }
  }
  PrintStderr(config, 1, "done\n")
  return r, nil
}

func ExportTrack(config SessionConfig, track Track, trackFilename string) error {
  PrintStderr(config, 1, "Writing track `%s'... ", trackFilename)
  parameters := DefaultBigWigParameters()
  parameters.ReductionLevels = config.BWZoomLevels
  if err := (GenericTrack{track}).ExportBigWig(trackFilename, parameters); err != nil {
    PrintStderr(config, 1, "failed\n")
    return err
  } else {
    PrintStderr(config, 1, "done\n")
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func ImportSingleTrackData(config SessionConfig, t ScalarType, filenameBw string, regions GRanges) ([]Vector, error) {
  n := regions.Length()
  r := make([]Vector, n)
  s := BinMean

  if t, err := config.GetBinSummaryStatistics(); err != nil {
    return nil, err
  } else {
    s = t
  }

  PrintStderr(config, 1, "Opening track file `%s'... ", filenameBw)
  // create a reader for each bigWig file
  f, err := os.Open(filenameBw)
  if err != nil {
    PrintStderr(config, 1, "failed\n")
    return nil, err
  }
  defer f.Close()

  bwr, err := NewBigWigReader(f); if err != nil {
    PrintStderr(config, 1, "failed\n")
    return nil, err
  }
  PrintStderr(config, 1, "done\n")

  PrintStderr(config, 1, "Importing data... ")
  for i := 0; i < n; i++ {
    if slice, _, err := bwr.QuerySlice(regions.Seqnames[i], regions.Ranges[i].From, regions.Ranges[i].To, s, config.BinSize, config.BinOverlap, config.TrackInit); err != nil {
      PrintStderr(config, 1, "failed\n")
      return nil, err
    } else {
      v := NullDenseVector(t, len(slice))
      for k := 0; k < len(slice); k++ {
        v.At(k).SetFloat64(slice[k])
      }
      r[i] = v
    }
  }
  PrintStderr(config, 1, "done\n")

  return r, nil
}

func ImportMultiTrackData(config SessionConfig, t ScalarType, filenamesBw []string, regions GRanges) ([]Matrix, error) {
  n   := regions.Length()
  m   := len(filenamesBw)
  bwr := make([]*BigWigReader, m)
  r   := make([]Matrix,        n)
  s   := BinMean

  if t, err := config.GetBinSummaryStatistics(); err != nil {
    return nil, err
  } else {
    s = t
  }

  // create a reader for each bigWig file
  for j, filename := range filenamesBw {
    PrintStderr(config, 1, "Opening track file `%s'... ", filename)
    f, err := os.Open(filename)
    if err != nil {
      PrintStderr(config, 1, "failed\n")
      return nil, err
    }
    defer f.Close()

    if reader, err := NewBigWigReader(f); err != nil {
      PrintStderr(config, 1, "failed\n")
      return nil, err
    } else {
      bwr[j] = reader
    }
    PrintStderr(config, 1, "done\n")
  }

  PrintStderr(config, 1, "Importing data... ")
  for i := 0; i < n; i++ {
    values  := []float64{}
    binSize := config.BinSize
    for j := 0; j < m; j++ {
      if slice, bs, err := bwr[j].QuerySlice(regions.Seqnames[i], regions.Ranges[i].From, regions.Ranges[i].To, s, binSize, config.BinOverlap, config.TrackInit); err != nil {
        return nil, err
      } else {
        values = append(values, slice...)
        if binSize == 0 {
          binSize = bs
        }
      }
    }
    if len(values) % m != 0 {
      PrintStderr(config, 1, "failed\n")
      return nil, fmt.Errorf("received data of varying lengths from bigWig files")
    }
    v := NullDenseVector(t, len(values))
    for k := 0; k < len(values); k++ {
      v.At(k).SetFloat64(values[k])
    }
    r[i] = v.AsMatrix(m, len(values)/m)
  }
  PrintStderr(config, 1, "done\n")

  return r, nil
}
