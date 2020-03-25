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
import   "math"
import   "bufio"
import   "bytes"
import   "compress/gzip"
import   "io"
import   "io/ioutil"
import   "os"
import   "strconv"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/utility"

import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/autodiff/statistics/generic"

/* -------------------------------------------------------------------------- */

func getNColors(n int) []string {
  // number of levels for each color
  k := int(math.Cbrt(float64(n)))+1
  // delta for each color
  d := 255/k
  s := []string{}
  for i := 0; i < n; i++ {
    r := (i/k/k % k) * d
    g := (i/k   % k) * d
    b := (i     % k) * d
    s  = append(s, fmt.Sprintf("%d,%d,%d", r,g,b))
  }
  return s
}

/* -------------------------------------------------------------------------- */

func ExportTrackSegmentation(config SessionConfig, track Track, bedFilename, bedName, bedDescription string, compress bool, stateNames, rgbChart []string, scores []Track) error {
  return (GenericTrack{track}).ExportSegmentation(bedFilename, bedName, bedDescription, compress, stateNames, rgbChart, scores)
}

/* -------------------------------------------------------------------------- */

func importTrackSegmentation(filename string) (GRanges, error) {
  var r io.Reader
  var g GRanges
  // open file
  f, err := os.Open(filename)
  if err != nil {
    return g, err
  }
  defer f.Close()
  // check if file is gzipped
  if IsGzip(filename) {
    gz, err := gzip.NewReader(f)
    if err != nil {
      return g, err
    }
    defer gz.Close()
    r = gz
  } else {
    r = f
  }
  // skip track line
  reader := bufio.NewReader(r)
  reader.ReadLine()
  // parse remaining file
  if err := g.ReadBed9(reader); err != nil {
    return g, err
  } else {
    return g, nil
  }
}

func ImportTrackSegmentation(config SessionConfig, bedFilename string, genome Genome, stateMap map[string]int) (Track, error) {
  var s TrackMutableSequence
  if r, err := importTrackSegmentation(bedFilename); err != nil {
    return nil, err
  } else {
    track  := AllocSimpleTrack("", genome, config.BinSize)
    states := r.GetMetaStr("name")

    if len(states) != r.Length() {
      return nil, fmt.Errorf("invalid segmentation bed file: name column is missing")
    }
    for i := 0; i < r.Length(); i++ {
      seqname := r.Seqnames[i]
      from    := r.Ranges  [i].From
      to      := r.Ranges  [i].To
      if i == 0 || r.Seqnames[i-1] != r.Seqnames[i] {
        if s_, err := track.GetMutableSequence(seqname); err != nil {
          return nil, err
        } else {
          s = s_
        }
      }
      if stateMap == nil {
        for k := from; k < to; k += config.BinSize {
          if len(states[i]) <= 1 {
            return nil, fmt.Errorf("invalid state at line %d", i+2)
          }
          value, err := strconv.ParseInt(states[i][1:], 10, 64); if err != nil {
            return nil, err
          }
          s.Set(k, float64(value))
        }
      } else {
        for k := from; k < to; k += config.BinSize {
          if value, ok := stateMap[states[i]]; !ok {
            return nil, fmt.Errorf("state `%s' not found in state map", states[i])
          } else {
            s.Set(k, float64(value))
          }
        }
      }
    }
    return track, nil
  }
}

/* -------------------------------------------------------------------------- */

func exportTrackSegmentation(granges GRanges, bedFilename, name, description string, compress bool) error {
  buffer := new(bytes.Buffer)

  w := bufio.NewWriter(buffer)
  if _, err := fmt.Fprintf(w, "track name=\"%s\" description=\"%s\" visibility=1 itemRgb=\"On\"\n", name, description); err != nil {
    return err
  }
  if err := granges.WriteBed9(w); err != nil {
    return err
  }
  w.Flush()

  if compress {
    b := new(bytes.Buffer)
    w := gzip.NewWriter(b)
    io.Copy(w, buffer)
    w.Close()
    buffer = b
  }
  return ioutil.WriteFile(bedFilename, buffer.Bytes(), 0666)
}

func ExportHierarchicalTrackSegmentation(config SessionConfig, track Track, bedFilename, bedName, bedDescription string, compress bool, stateNames, rgbChart []string, tree generic.HmmNode, level int) error {
  r, err := GenericTrack{track}.GRanges("state"); if err != nil {
    return err
  }
  rgbMap := make(map[int]int)
  rgbCnt := 0
  var f func(node generic.HmmNode, d int) ([]int, error)
  f = func(node generic.HmmNode, d int) ([]int, error) {
    if d == level {
      if node.Children == nil {
        for i := 0; i < len(node.Children); i++ {
          for k := node.States[0]; k < node.States[1]; k++ {
            rgbMap[k] = rgbCnt
          }
          rgbCnt++
        }
      } else {
        for i := 0; i < len(node.Children); i++ {
          if r, err := f(node.Children[i], d+1); err != nil {
            return nil, err
          } else {
            for _, k := range r {
              rgbMap[k] = rgbCnt
            }
          }
          rgbCnt++
        }
      }
      return nil, nil
    } else {
      states := []int{}
      if node.Children == nil {
        for k := node.States[0]; k < node.States[1]; k++ {
          states = append(states, k)
        }
      } else {
        for i := 0; i < len(node.Children); i++ {
          if r, err := f(node.Children[i], d+1); err != nil {
            return nil, err
          } else {
            states = append(states, r...)
          }
        }
      }
      return states, nil
    }
  }
  if _, err := f(tree, 0); err != nil {
    return err
  }
  if len(rgbMap) == 0 {
    return fmt.Errorf("invalid level")
  }
  if len(stateNames) == 0 {
    stateNames = make([]string, len(rgbMap))
    for i := 0; i < len(stateNames); i++ {
      stateNames[i] = fmt.Sprintf("s%d", i)
    }
  }
  if len(rgbChart) == 0 {
    rgbChart = getNColors(len(rgbMap))
  }
  if len(rgbMap) != len(stateNames) {
    return fmt.Errorf("invalid number of state names")
  }
  // get states for the lowest level
  state_old  := r.GetMetaFloat("state"); if len(state_old) == 0 {
    return nil
  }
  // convert states to higher level states
  for i := 0; i < len(state_old); i++ {
    // convert score to state
    s := int(state_old[i])
    if s < 0 || math.Floor(state_old[i]) != state_old[i] {
      return fmt.Errorf("invalid state `%f' at `%s:%d-%d", state_old[i], r.Seqnames[i], r.Ranges[i].From, r.Ranges[i].To)
    }
    if t, ok := rgbMap[s]; !ok {
      return fmt.Errorf("rgbChart has not enough colors; invalid level or tree")
    } else {
      state_old[i] = float64(t)
    }
  }
  // merge consecutive bins with the same state
  seqnames   := []string {r.Seqnames[0]}
  from       := []int    {r.Ranges  [0].From}
  to         := []int    {r.Ranges  [0].To}
  state      := []float64{state_old [0]}

  for i := 1; i < r.Length(); i++ {
    if r.Seqnames[i-1] == r.Seqnames[i] && r.Ranges[i-1].To == r.Ranges[i].From && state_old[i-1] == state_old[i] {
      to[len(to)-1] = r.Ranges[i].To
    } else {
      seqnames = append(seqnames, r.Seqnames[i])
      from     = append(from,     r.Ranges  [i].From)
      to       = append(to,       r.Ranges  [i].To)
      state    = append(state,    state_old [i])
    }
  }
  r = NewGRanges(seqnames, from, to, nil)

  // add color information as meta columns
  name       := make([]string, len(state))
  score      := make([]int,    len(state))
  thickStart := make([]int,    len(state))
  thickEnd   := make([]int,    len(state))
  itemRgb    := make([]string, len(state))

  for i := 0; i < r.Length(); i++ {
    name      [i] = stateNames[int(state[i])]
    thickStart[i] = r.Ranges[i].From
    thickEnd  [i] = r.Ranges[i].To
    itemRgb   [i] = rgbChart[int(state[i])]
    score     [i] = 0.0
  }
  r.AddMeta("name",       name)
  r.AddMeta("score",      score)
  r.AddMeta("thickStart", thickStart)
  r.AddMeta("thickEnd",   thickEnd)
  r.AddMeta("itemRgb",    itemRgb)
  // write result to file
  return exportTrackSegmentation(r, bedFilename, bedName, bedDescription, compress)
}
