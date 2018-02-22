
package main

/* -------------------------------------------------------------------------- */

//import "fmt"
import "log"
import "math/rand"
import "os"

import . "github.com/pbenner/ngstat/config"
//import . "github.com/pbenner/ngstat/classification"
import . "github.com/pbenner/ngstat/estimation"
//import . "github.com/pbenner/ngstat/track"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/generic"
import   "github.com/pbenner/autodiff/statistics/scalarEstimator"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

//import . "github.com/pbenner/gonetics"
//import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func newEstimator(n int) VectorEstimator {
  components := make([]ScalarEstimator, n)
  {
    if delta, err := scalarEstimator.NewDeltaEstimator(0.0); err != nil {
      panic(err)
    } else {
      components[0] = delta
    }
  }
  for i := 1; i < n; i++ {
    if poisson, err := scalarEstimator.NewPoissonEstimator(rand.Float64()); err != nil {
      panic(err)
    } else {
      components[i] = poisson
    }
  }
  if mixture, err := scalarEstimator.NewMixtureEstimator(nil, components, 1e-8, -1, generic.DefaultEmHook(os.Stdout)); err != nil {
    panic(err)
  } else {
    if estimator, err := vectorEstimator.NewScalarIid(mixture, -1); err != nil {
      panic(err)
    } else {
      return estimator
    }
  }
}

/* -------------------------------------------------------------------------- */

func learnModel(config SessionConfig, filenameIn string, n_components int) VectorPdf {

  estimator := newEstimator(n_components)

  if err := ImportAndEstimateOnSingleTrack(config, estimator, filenameIn, "chr18"); err != nil {
    panic(err)
  }

  return estimator.GetEstimate()
}

/* -------------------------------------------------------------------------- */

func LearnModel(config SessionConfig, args []string) {

  if len(args) != 2 {
    log.Fatal("Usage: LearnModel <OUTPUT.json> <INPUT.bw>")
  }

  filenameOut := args[0]
  filenameIn  := args[1]

  ExportDistribution(filenameOut,
    learnModel(config, filenameIn, 3))

}
