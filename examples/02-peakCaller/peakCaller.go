
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
  // if d, err := scalarDistribution.NewNegativeBinomialDistribution(NewReal(1.0), NewReal(0.5)); err != nil {
  //   log.Fatal(err)
  // } else {
  //   if numeric, err := scalarEstimator.NewNumericEstimator(d); err != nil {
  //     log.Fatal(err)
  //   } else {
  //     numeric.MaxIterations = 5
  //     numeric.Epsilon = 1e-2
  //     numeric.Method = "bfgs"
  //     numeric.Hook = func(variables ConstVector, r ConstScalar) error {
  //       fmt.Println(" -> numeric optimization:", variables, r)
  //       return nil
  //     }
  //     components = append(components, numeric)
  //   }
  // }
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

func callPeaks(config SessionConfig, filenameOut, filenameIn1, filenameIn2 string, mixture1, mixture2 *scalarDistribution.Mixture, k int) MutableTrack {

  scalarClassifier1 := scalarClassifier.MixturePosterior{mixture1, []int{k}}
  vectorClassifier1 := vectorClassifier.ScalarBatchIid{scalarClassifier1, 1}
  scalarClassifier2 := scalarClassifier.MixturePosterior{mixture2, []int{k}}
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

  optComponent      := options.   IntLong("component",       0,   3, "foreground mixture component")
  optModelTreatment := options.StringLong("model-treatment", 0,  "", "json file containing the treatment mixture model")
  optModelControl   := options.StringLong("model-control",   0,  "", "json file containing the control mixture model")

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

  if *optModelTreatment != "" {
    if t, err := ImportScalarPdf(*optModelTreatment, BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model1 = t.(*scalarDistribution.Mixture)
    }
  } else {
    model1 = learnModel(config, filenameIn1)
  }
  if *optModelControl != "" {
    if t, err := ImportScalarPdf(*optModelControl, BareRealType); err != nil {
      log.Fatal(err)
    } else {
      model2 = t.(*scalarDistribution.Mixture)
    }
  } else {
    model2 = learnModel(config, filenameIn2)
  }

  result := callPeaks(config, filenameOut, filenameIn1, filenameIn2, model1, model2, *optComponent)

  if err := ExportTrack(config, result, filenameOut); err != nil {
    log.Fatal(err)
  }
}
