
package main

/* -------------------------------------------------------------------------- */

//import "fmt"
import "log"
import "math"
import "math/rand"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/classification"
import . "github.com/pbenner/ngstat/estimation"
import . "github.com/pbenner/ngstat/track"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/scalarClassifier"
import   "github.com/pbenner/autodiff/statistics/scalarDistribution"
import   "github.com/pbenner/autodiff/statistics/scalarEstimator"
import   "github.com/pbenner/autodiff/statistics/vectorClassifier"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

var ConfigFilename = "config.json"

/* -------------------------------------------------------------------------- */

func newEstimator(config SessionConfig) VectorEstimator {
  components := make([]ScalarEstimator, 4)
  {
    if delta, err := scalarEstimator.NewDeltaEstimator(0.0); err != nil {
      panic(err)
    } else {
      components[0] = delta
    }
  }
  {
    if poisson, err := scalarEstimator.NewPoissonEstimator(rand.Float64()); err != nil {
      panic(err)
    } else {
      components[1] = poisson
    }
  }
  {
    if poisson, err := scalarEstimator.NewGeometricEstimator(rand.Float64()); err != nil {
      panic(err)
    } else {
      components[2] = poisson
    }
  }
  {
    if poisson, err := scalarEstimator.NewGeometricEstimator(rand.Float64()); err != nil {
      panic(err)
    } else {
      components[3] = poisson
    }
  }
  if mixture, err := scalarEstimator.NewMixtureEstimator(nil, components, 1e-8, -1); err != nil {
    panic(err)
  } else {
    // set options
    mixture.Verbose = config.Verbose
    if estimator, err := vectorEstimator.NewScalarIid(mixture, -1); err != nil {
      panic(err)
    } else {
      return estimator
    }
  }
}

/* -------------------------------------------------------------------------- */

func learnModel(config SessionConfig, filenameIn string) *scalarDistribution.Mixture {

  estimator := newEstimator(config)

  if err := ImportAndEstimateOnSingleTrack(config, estimator, filenameIn); err != nil {
    panic(err)
  }

  return estimator.GetEstimate().(*vectorDistribution.ScalarIid).Distribution.(*scalarDistribution.Mixture)
}

func callPeaks(config SessionConfig, filenameOut, filenameIn string, mixture *scalarDistribution.Mixture) MutableTrack {

  scalarClassifier := scalarClassifier.MixturePosterior{mixture, []int{3}}
  vectorClassifier := vectorClassifier.ScalarBatchIid{scalarClassifier, 1}

  if result, err := ImportAndBatchClassifySingleTrack(config, vectorClassifier, filenameIn); err != nil {
    log.Fatal(err)
  } else {
    if err := (GenericMutableTrack{result}).Map(result, func(seqname string, position int, value float64) float64 {
      return math.Exp(value)
    }); err != nil {
      log.Fatal(err)
    }
    return result
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func LearnModel(config SessionConfig, args []string) {

  if len(args) != 2 {
    log.Fatal("Usage: LearnModel <OUTPUT.json> <INPUT.bw>")
  }

  filenameOut := args[0]
  filenameIn  := args[1]

  ExportDistribution(filenameOut,
    learnModel(config, filenameIn))

}

/* -------------------------------------------------------------------------- */

func CallPeaks(config SessionConfig, args []string) {

  if len(args) != 2 && len(args) != 3 {
    log.Fatal("Usage: LearnModel <OUTPUT.bw> <INPUT.bw> [MODEL.json]")
  }
  var model *scalarDistribution.Mixture

  filenameOut := args[0]
  filenameIn  := args[1]

  if len(args) == 3 {
    if t, err := ImportScalarPdf(args[2], BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model = t.(*scalarDistribution.Mixture)
    }
  } else {
    model = learnModel(config, filenameIn)
  }

  result := callPeaks(config, filenameOut, filenameIn, model)

  if err := ExportTrack(config, result, filenameOut); err != nil {
    log.Fatal(err)
  }
}
