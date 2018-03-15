
package main

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "log"
import   "math"
import   "math/rand"
import   "os"
import   "strconv"
import   "strings"

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
  components := []ScalarEstimator{}
  for i := 0; i <= 6; i++ {
    if delta, err := scalarEstimator.NewDeltaEstimator(float64(i)); err != nil {
      log.Fatal(err)
    } else {
      components = append(components, delta)
    }
  }
  if poisson, err := scalarEstimator.NewPoissonEstimator(rand.Float64()); err != nil {
    log.Fatal(err)
  } else {
    if t, err := scalarEstimator.NewTranslationEstimator(poisson, -5.0); err != nil {
      log.Fatal(err)
    } else {
      components = append(components, t)
    }
  }
  if poisson, err := scalarEstimator.NewGeometricEstimator(0.02); err != nil {
    log.Fatal(err)
  } else {
    components = append(components, poisson)
  }
  if mixture, err := scalarEstimator.NewDiscreteMixtureEstimator(nil, components, 1e-6, -1); err != nil {
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

func callPeaks(config SessionConfig, filenameOut, filenameIn1, filenameIn2 string, mixture1, mixture2 *scalarDistribution.Mixture, k1, k2 []int) MutableTrack {

  scalarClassifier1 := scalarClassifier.MixturePosterior{mixture1, k1}
  vectorClassifier1 := vectorClassifier.ScalarBatchIid{scalarClassifier1, 1}
  scalarClassifier2 := scalarClassifier.MixturePosterior{mixture2, k2}
  vectorClassifier2 := vectorClassifier.ScalarBatchIid{scalarClassifier2, 1}

  result1, err := ImportAndBatchClassifySingleTrack(config, vectorClassifier1, filenameIn1); if err != nil {
    log.Fatal(err)
  }
  result2, err := ImportAndBatchClassifySingleTrack(config, vectorClassifier2, filenameIn2); if err != nil {
    log.Fatal(err)
  }
  if err := (GenericMutableTrack{result1}).MapList([]Track{result1, result2}, func(seqname string, position int, values... float64) float64 {
    return math.Exp(values[0])*(1.0-math.Exp(values[1]))
  }); err != nil {
    log.Fatal(err)
  }
  return result1
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

  optComponents1  := options.StringLong("components-treatment", 0, "2,3", "treatment foreground mixture component")
  optComponents2  := options.StringLong("components-control",   0,   "8", "control foreground mixture component")
  optModel1       := options.StringLong("model-treatment",      0,    "", "json file containing the treatment mixture model")
  optModel2       := options.StringLong("model-control",        0,    "", "json file containing the control mixture model")

  options.SetParameters("<OUTPUT.bw> <TREATMENT.bw> <CONTROL.bw>")
  options.Parse(append([]string{"LearnModel"}, args...))

  if len(options.Args()) != 3 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  filenameOut := options.Args()[0]
  filenameIn1 := options.Args()[1]
  filenameIn2 := options.Args()[2]

  var model1 *scalarDistribution.Mixture
  var model2 *scalarDistribution.Mixture

  var components1 []int
  var components2 []int

  if *optModel1 != "" {
    if t, err := ImportScalarPdf(*optModel1, BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model1 = t.(*scalarDistribution.Mixture)
    }
  } else {
    model1 = learnModel(config, filenameIn1)
  }
  if *optModel1 != "" {
    if t, err := ImportScalarPdf(*optModel2, BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model2 = t.(*scalarDistribution.Mixture)
    }
  } else {
    model2 = learnModel(config, filenameIn2)
  }
  for _, str := range strings.Split(*optComponents1, ",") {
    if i, err := strconv.ParseInt(str, 10, 64); err != nil {
      log.Fatal(err)
    } else {
      components1 = append(components1, int(i))
    }
  }
  for _, str := range strings.Split(*optComponents2, ",") {
    if i, err := strconv.ParseInt(str, 10, 64); err != nil {
      log.Fatal(err)
    } else {
      components2 = append(components2, int(i))
    }
  }
  if len(components1) == 0 {
    log.Fatal("empty set of treatment components")
  }
  if len(components2) == 0 {
    log.Fatal("empty set of control components")
  }

  result := callPeaks(config, filenameOut, filenameIn1, filenameIn2, model1, model2, components1, components2)

  if err := ExportTrack(config, result, filenameOut); err != nil {
    log.Fatal(err)
  }
}
