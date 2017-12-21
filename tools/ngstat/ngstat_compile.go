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

import   "fmt"
import   "log"
import   "os"
import   "os/exec"

import . "github.com/pbenner/ngstat/config"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func ngstat_compile_main(config SessionConfig, args []string) {

  options := getopt.New()
  options.SetProgram(fmt.Sprintf("%s statistics", os.Args[0]))

  optHelp := options.   BoolLong("help",     'h',     "print help")

  options.SetParameters("<PLUGIN.go>\n")
  options.Parse(args)

  // command options
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // command arguments
  if len(options.Args()) < 1 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }

  filename := options.Args()[0]

  cmd := exec.Command("go", "build", "-buildmode=plugin", "-i", "-v", filename)
  cmd.Stdout = os.Stdout
  cmd.Stderr = os.Stderr

  if err := cmd.Run(); err != nil {
    log.Fatal(err)
  }
}
