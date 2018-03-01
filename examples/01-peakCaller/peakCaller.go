
package main

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "log"
import   "math"
import   "math/rand"
import   "os"

import   "github.com/pborman/getopt"

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
  if delta, err := scalarEstimator.NewDeltaEstimator(0.0); err != nil {
    log.Fatal(err)
  } else {
    components[0] = delta
  }
  if poisson, err := scalarEstimator.NewPoissonEstimator(rand.Float64()); err != nil {
    log.Fatal(err)
  } else {
    components[1] = poisson
  }
  if poisson, err := scalarEstimator.NewGeometricEstimator(rand.Float64()); err != nil {
    log.Fatal(err)
  } else {
    components[2] = poisson
  }
  if poisson, err := scalarEstimator.NewGeometricEstimator(rand.Float64()); err != nil {
    log.Fatal(err)
  } else {
    components[3] = poisson
  }
  if mixture, err := scalarEstimator.NewMixtureEstimator(nil, components, 1e-8, -1); err != nil {
    log.Fatal(err)
  } else {
    // set options
    mixture.Verbose = config.Verbose
    if estimator, err := vectorEstimator.NewScalarIid(mixture, -1); err != nil {
      log.Fatal(err)
    } else {
      return estimator
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func learnModel(config SessionConfig, filenameIn string) *scalarDistribution.Mixture {

  estimator := newEstimator(config)

  if err := ImportAndEstimateOnSingleTrack(config, estimator, filenameIn); err != nil {
    log.Fatal(err)
  }

  return estimator.GetEstimate().(*vectorDistribution.ScalarIid).Distribution.(*scalarDistribution.Mixture)
}

func callPeaks(config SessionConfig, filenameOut, filenameIn string, mixture *scalarDistribution.Mixture, k int) MutableTrack {

  scalarClassifier := scalarClassifier.MixturePosterior{mixture, []int{k}}
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

  options := getopt.New()

  optComponent := options.   IntLong("component", 0,   3, "foreground mixture component")
  optModel     := options.StringLong("model",     0,  "", "json file containing the mixture model")

  options.SetParameters("<OUTPUT.bw> <INPUT.bw>")
  options.Parse(append([]string{"LearnModel"}, args...))

  if len(options.Args()) != 2 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  filenameOut := options.Args()[0]
  filenameIn  := options.Args()[1]

  var model *scalarDistribution.Mixture

  if *optModel != "" {
    if t, err := ImportScalarPdf(*optModel, BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model = t.(*scalarDistribution.Mixture)
    }
  } else {
    model = learnModel(config, filenameIn)
  }

  result := callPeaks(config, filenameOut, filenameIn, model, *optComponent)

  if err := ExportTrack(config, result, filenameOut); err != nil {
    log.Fatal(err)
  }
}
