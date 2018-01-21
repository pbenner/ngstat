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

package main

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "log"
import   "os"
import   "strconv"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func main() {

  options := getopt.New()

  optConfig    := options. StringLong("config",        'c',     "", "configuration file")
  optBinSize   := options.    IntLong("bin-size",       0 ,      0, "bin size")
  optBinStat   := options. StringLong("bin-summary",    0 , "mean", "bin summary statistic [mean (default), max, min, discrete mean]")
  optBinOver   := options.    IntLong("bin-overlap",    0 ,      0, "number of overlapping bins when computing the summary")
  optTrackInit := options. StringLong("initial-value",  0 ,     "", "track initial value [default: 0]")
  optThreads   := options.    IntLong("threads",        0 ,      0, "number of threads")
  optHelp      := options.   BoolLong("help",          'h',         "print help")
  optVerbose   := options.CounterLong("verbose",       'v',         "verbose level [-v or -vv]")

  options.SetParameters("<COMMAND>\n\n" +
    " Commands:\n" +
    "     compile          - compile a plugin\n" +
    "     exec             - execute a plugin\n")
  options.Parse(os.Args)

  config := DefaultSessionConfig()

  // command options
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optVerbose != 0 {
    config.Verbose = *optVerbose
  }
  if *optConfig != "" {
    current_config := config
    PrintStderr(current_config, 1, "Importing config file `%s'... ", *optConfig)
    if err := config.ImportFile(*optConfig); err != nil {
      PrintStderr(current_config, 1, "failed\n")
      log.Fatalf("reading config file `%s' failed: %v", *optConfig, err)
    }
    PrintStderr(current_config, 1, "done\n")
  }
  if options.Lookup("bin-size").Seen() {
    if *optBinSize < 0 {
      log.Fatalf("invalid bin-size `%d'", *optBinSize)
    }
    config.BinSize = *optBinSize
  }
  if options.Lookup("bin-summary").Seen() {
    config.BinSummaryStatistics = *optBinStat
  }
  if options.Lookup("bin-overlap").Seen() {
    config.BinOverlap = *optBinOver
  }
  if options.Lookup("initial-value").Seen() {
    v, err := strconv.ParseFloat(*optTrackInit, 64)
    if err != nil {
      log.Fatalf("parsing initial value failed: %v", err)
    }
    config.TrackInit = v
  }
  if options.Lookup("threads").Seen() {
    if *optThreads < 1 {
      log.Fatalf("invalid number of threads `%d'", *optThreads)
    }
    config.Threads = *optThreads
  }
  // command arguments
  if len(options.Args()) == 0 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  command := options.Args()[0]

  switch command {
  case "compile":
    ngstat_compile_main(config, options.Args())
  case "exec":
    ngstat_exec_main(config, options.Args())
  default:
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
}
