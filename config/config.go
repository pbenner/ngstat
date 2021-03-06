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

package config

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "bytes"
import   "io"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type SessionConfig struct {
  Threads                int
  Verbose                int
  BinSummaryStatistics   string  `json:"Bin Summary Statistics"`
  BWZoomLevels         []int     `json:"BigWig Zoom Levels"`
  BinSize                int     `json:"Bin Size"`
  BinOverlap             int     `json:"Bin Overlap"`
  TrackInit              float64 `json:"Track Initial Value"`
}

func (config *SessionConfig) Import(reader io.Reader, args... interface{}) error {
  return JsonImport(reader, config)
}

func (config *SessionConfig) Export(writer io.Writer) error {
  return JsonExport(writer, config)
}

func (config *SessionConfig) ImportFile(filename string) error {
  if err := ImportFile(config, filename, Float64Type); err != nil {
    return err
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func DefaultSessionConfig() SessionConfig {
  config := SessionConfig{}
  // set default values
  config.BinSummaryStatistics = "mean"
  config.BWZoomLevels         = nil   // zoom levels are determined automatically
  config.BinSize              = 0
  config.BinOverlap           = 0
  config.TrackInit            = 0
  config.Threads              = 1
  return config
}

/* -------------------------------------------------------------------------- */

func (config *SessionConfig) GetBinSummaryStatistics() (BinSummaryStatistics, error) {
  if s := BinSummaryStatisticsFromString(config.BinSummaryStatistics); s == nil {
    return nil, fmt.Errorf("invalid bin summary statistics: %s", config.BinSummaryStatistics)
  } else {
    return s, nil
  }
}

/* -------------------------------------------------------------------------- */

func (config *SessionConfig) String() string {
  var buffer bytes.Buffer

  fmt.Fprintf(&buffer, "Session Config:\n")
  fmt.Fprintf(&buffer, " -> Bin Overlap            : %v\n", config.BinOverlap)
  fmt.Fprintf(&buffer, " -> Bin Summary Statistics : %v\n", config.BinSummaryStatistics)
  fmt.Fprintf(&buffer, " -> Bin Size               : %v\n", config.BinSize)
  fmt.Fprintf(&buffer, " -> BigWig Zoom Levels     : %v\n", config.BWZoomLevels)
  fmt.Fprintf(&buffer, " -> Track Initial Value    : %v\n", config.TrackInit)
  fmt.Fprintf(&buffer, " -> Threads                : %v\n", config.Threads)
  fmt.Fprintf(&buffer, " -> Verbose                : %v\n", config.Verbose)

  return buffer.String()
}
