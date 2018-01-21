/* Copyright (C) 2018 Philipp Benner
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

package main

/* -------------------------------------------------------------------------- */

import   "log"
import   "strconv"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/estimation"
import   "github.com/pbenner/ngstat/statistics/nonparametric"
import . "github.com/pbenner/ngstat/track"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

/* -------------------------------------------------------------------------- */

func Estimate(config SessionConfig, args []string) {
  if len(args) != 3 {
    log.Println("Usage: Estimate <NBINS> <INPUT.bw> <OUTPUT.json>")
    log.Fatal("invalid number of arguments")
  }
  bins, err := strconv.ParseInt(args[0], 10, 64); if err != nil {
    log.Fatal(err)
  }
  filenameIn  := args[1]
  filenameOut := args[2]

  estimator0, _ :=   nonparametric.NewEstimator(int(bins), 0.0)
  estimator , _ := vectorEstimator.NewScalarBatchId(estimator0)

  track, err := ImportTrack(config, filenameIn); if err != nil {
    panic(err)
  }
  if err := BatchEstimateOnSingleTrack(config, estimator, track, -1); err != nil {
    panic(err)
  }
  if err := ExportDistribution(filenameOut, estimator.GetEstimate()); err != nil {
    panic(err)
  }
}
