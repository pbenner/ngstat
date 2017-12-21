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
import   "plugin"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func ngstat_exec_load_config(config *SessionConfig, plugin *plugin.Plugin) {
  f, err := plugin.Lookup("ConfigFilename")
  if err == nil {
    switch filename := f.(type) {
    case *string:
      current_config := *config
      PrintStderr(current_config, 1, "Importing config file `%s'... ", *filename)
      if err := config.ImportFile(*filename); err != nil {
        PrintStderr(current_config, 1, "failed\n")
        log.Fatalf("reading config file `%s' failed: %v", *filename, err)
      }
      PrintStderr(current_config, 1, "done\n")
    default:
      log.Fatal("error while reading config filename: variable has invalid type")
    }
  }
  c, err := plugin.Lookup("ConfigVariable")
  if err == nil {
    switch configPtr := c.(type) {
    case *SessionConfig:
      PrintStderr(*config, 1, "Importing config from plugin.\n")
      *config = *configPtr
    default:
      log.Fatal("error while reading config: variable has invalid type")
    }
  }
}

/* -------------------------------------------------------------------------- */

func ngstat_exec_generic_main(config SessionConfig, args []string, plugin *plugin.Plugin, fname string) {
  g, err := plugin.Lookup(fname)
  if err != nil {
    log.Fatal(err)
  }
  switch f := g.(type) {
  case func(config SessionConfig, args []string):
    f(config, args)
  case func(config SessionConfig):
    f(config)
  case func(args []string):
    f(args)
  case func():
    f()
  default:
    log.Fatal("error while executing plugin: function has invalid type")
  }
}

/* -------------------------------------------------------------------------- */

func ngstat_exec_main(config SessionConfig, args []string) {

  options := getopt.New()
  options.SetProgram(fmt.Sprintf("%s statistics", os.Args[0]))

  optHelp := options.   BoolLong("help",     'h',     "print help")

  options.SetParameters("<PLUGIN.so> <FUNCTION_NAME>\n")
  options.Parse(args)

  // command options
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // command arguments
  if len(options.Args()) < 2 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }

  filename := options.Args()[0]
  command  := options.Args()[1]

  p, err := plugin.Open(filename)
  if err != nil {
    log.Fatalf("opening plugin `%s' failed: %v", filename, err)
  }

  ngstat_exec_load_config(&config, p)
  ngstat_exec_generic_main(config, options.Args(), p, command)
}
