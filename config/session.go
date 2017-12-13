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
import   "io"
import   "path/filepath"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type SessionConfig struct {
  Align                  bool
  AlignMaxShift          int
  Threads                int
  Verbose                int
  BinSummaryStatistics   string  `json:"Bin Summary Statistics"`
  BWZoomLevels         []int     `json:"BigWig Zoom Levels"`
  FastaPath              string  `json:"Fasta Path"`
  WindowSize             int     `json:"Window Size"`
  BinSize                int     `json:"Bin Size"`
  BinOverlap             int     `json:"Bin Overlap"`
  TrackInit              float64 `json:"Track Initial Value"`
  Fraglen                int     `json:"Fragment Length"`
  FraglenRange        [2]int     `json:"Fragment Length Range"`
  EstimateFraglen        bool    `json:"Estimate Read Extension"`
  PairedEnd              bool    `json:"Paired-End Reads"`
  FeasibleReadLengths [2]int     `json:"Feasible Read Lengths"`
  MinMapQ                int     `json:"Minimum Mapping Quality"`
  RmDup                  bool    `json:"Remove Duplicates"`
  BinningMethod          string  `json:"Binning Method"`
  NormalizeTrack         string  `json:"Normalize Track"`
  FilterStrand           byte    `json:"Filter Strand"`
  ShiftReads          [2]int     `json:"Shift Reads"`
  LogScale               bool    `json:"Log Scale"`
  Pseudocounts        [2]float64 `json:"Pseudocounts"`
  TrackPath              string  `json:"Track Path"`
  GenomePath             string  `json:"Genome Path"`
  SmoothenControl        bool    `json:"Smoothen Control"`
  SmoothenSizes        []int     `json:"Smoothen Window Sizes"`
  SmoothenMin            float64 `json:"Smoothen Minimum Counts"`
}

func (config *SessionConfig) Import(reader io.Reader, args... interface{}) error {
  return JsonImport(reader, config)
}

func (config *SessionConfig) Export(writer io.Writer) error {
  return JsonExport(writer, config)
}

func (config *SessionConfig) ImportFile(filename string) error {
  if err := ImportFile(config, filename, BareRealType); err != nil {
    return err
  }
  // if filename is not in the current directory, change all relative paths
  if path := filepath.Dir(filename); path != "." {
    if config.FastaPath != "" && !filepath.IsAbs(config.FastaPath) {
      config.FastaPath = filepath.Join(path, config.FastaPath)
    }
    if config.GenomePath != "" && !filepath.IsAbs(config.GenomePath) {
      config.GenomePath = filepath.Join(path, config.GenomePath)
    }
    if config.TrackPath != "" && !filepath.IsAbs(config.TrackPath) {
      config.TrackPath = filepath.Join(path, config.TrackPath)
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func DefaultSessionConfig() SessionConfig {
  config := SessionConfig{}
  // set default values
  config.Align                = false
  config.BinSummaryStatistics = "mean"
  config.BWZoomLevels         = nil   // zoom levels are determined automatically
  config.WindowSize           = 100
  config.BinSize              = 10
  config.BinOverlap           = 0
  config.TrackInit            = 0
  config.Fraglen              = 0
  config.FraglenRange         = [2]int{-1, -1}
  config.FeasibleReadLengths  = [2]int{0,0}
  config.MinMapQ              = 0
  config.RmDup                = false
  config.BinningMethod        = "overlap"
  config.FilterStrand         = '*'
  config.LogScale             = false
  config.Pseudocounts         = [2]float64{0.0, 0.0}
  config.SmoothenControl      = false
  config.SmoothenSizes        = []int{}
  config.SmoothenMin          = 20.0
  config.Threads              = 1
  return config
}

/* -------------------------------------------------------------------------- */

func (config *SessionConfig) GetGenomePath() (string, error) {
  if config.GenomePath == "" {
    return "", fmt.Errorf("no genome path specified")
  }
  return config.GenomePath, nil
}

func (config *SessionConfig) GetTrackPath() (string, error) {
  if config.TrackPath == "" {
    return "", fmt.Errorf("no track path specified")
  }
  return config.TrackPath, nil
}

func (config *SessionConfig) GetBinSummaryStatistics() (BinSummaryStatistics, error) {
  switch config.BinSummaryStatistics {
  case "mean":
    return BinMean, nil
  case "discrete mean":
    return BinDiscreteMean, nil
  case "min":
    return BinMin, nil
  case "max":
    return BinMax, nil
  }
  return nil, fmt.Errorf("invalid bin summary statistics")
}
