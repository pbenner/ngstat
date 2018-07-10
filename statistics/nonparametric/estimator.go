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

package nonparametric

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/logarithmetic"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/scalarEstimator"

import   "github.com/pbenner/smartBinning"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type NonparametricEstimator struct {
  *NonparametricDistribution
  scalarEstimator.StdEstimator
  MargCounts      map[float64]float64
  Dimension       int
  NBins           int
  MaxBins         int
  BySize          bool
  Verbose         bool
}

/* -------------------------------------------------------------------------- */

func NewEstimator(nbins int) (*NonparametricEstimator, error) {
  r := NonparametricEstimator{}
  r.NonparametricDistribution, _ = NullDistribution([]float64{})
  r.NBins         = nbins
  // restrict maximum number of bins for improving performance
  r.MaxBins       = 1000000
  r.BySize        = true
  r.Verbose       = false
  if r.NBins > r.MaxBins {
    r.NBins = r.MaxBins
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *NonparametricEstimator) Clone() *NonparametricEstimator {
  r, _ := NewEstimator(obj.NBins)
  r.NonparametricDistribution = obj.NonparametricDistribution.Clone()
  r.MaxBins = obj.MaxBins
  r.BySize  = obj.BySize
  r.Verbose = obj.Verbose
  if obj.MargCounts != nil {
    r.Initialize(threadpool.Nil())
    for k, v := range obj.MargCounts {
      r.MargCounts[k] = v
    }
  }
  return r
}

func (obj *NonparametricEstimator) CloneScalarEstimator() ScalarEstimator {
  return obj.Clone()
}

func (obj *NonparametricEstimator) CloneScalarBatchEstimator() ScalarBatchEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *NonparametricEstimator) histogramMax(values []float64) float64 {
  max1 := math.Inf(-1)
  max2 := math.Inf(-1)
  for i := 0; i < len(values); i++ {
    if values[i] > max1 {
      max2 = max1
      max1 = values[i]
    }
  }
  if !math.IsInf(max2, -1) {
    return max1 + (max1-max2)/2
  } else {
    return max1 + 1.0
  }
}

func (obj *NonparametricEstimator) filterBinsMax() {
  // find miminum and maximum value
  minimum := math.Inf( 1)
  maximum := math.Inf(-1)
  for k, _ := range obj.MargCounts {
    if k < minimum {
      minimum = k
    }
    if k > maximum {
      maximum = k
    }
  }
  if math.IsInf(minimum, 0) || math.IsNaN(minimum) {
    return
  }
  if math.IsInf(maximum, 0) || math.IsNaN(maximum) {
    return
  }
  delta  := (maximum - minimum)/float64(obj.MaxBins)
  counts := obj.MargCounts
  obj.MargCounts = make(map[float64]float64)
  // round all values
  for k1, v := range counts {
    k2 := math.Floor(k1/delta)*delta
    obj.MargCounts[k2] += v
  }
}

func (obj *NonparametricEstimator) computeBins() ([]float64, []float64) {
  var histogram *smartBinning.Binning
  // for improving performance, first bin data into MaxBins bins
  if len(obj.MargCounts) > obj.MaxBins {
    obj.filterBinsMax()
  }
  // collect all values
  values := []float64{}
  counts := []float64{}
  for x, c := range obj.MargCounts {
    values = append(values, x)
    counts = append(counts, c)
  }
  values = append(values, obj.histogramMax(values))
  if obj.BySize {
    histogram, _ = smartBinning.New(values, counts, smartBinning.BinLogSum, smartBinning.BinLessSize)
  } else {
    histogram, _ = smartBinning.New(values, counts, smartBinning.BinLogSum, smartBinning.BinLessY)
  }
  histogram.Verbose = obj.Verbose
  histogram.FilterBins(obj.NBins)
  values = []float64{}
  counts = []float64{}
  for i := 0; i < len(histogram.Bins); i++ {
    values = append(values, histogram.Bins[i].Lower)
    counts = append(counts, histogram.Bins[i].Y)
  }
  // transform counts map
  return values, counts
}

/* batch estimator interface
 * -------------------------------------------------------------------------- */

func (obj *NonparametricEstimator) Initialize(p threadpool.ThreadPool) error {
  obj.MargCounts = make(map[float64]float64)
  return nil
}

func (obj *NonparametricEstimator) NewObservation(x, gamma ConstScalar, p threadpool.ThreadPool) error {
  if math.IsNaN(x.GetValue()) {
    return nil
  }
  if gamma != nil {
    if r, ok := obj.MargCounts[x.GetValue()]; ok {
      obj.MargCounts[x.GetValue()] = LogAdd(r, gamma.GetValue())
    } else {
      obj.MargCounts[x.GetValue()] = gamma.GetValue()
    }
  } else {
    if r, ok := obj.MargCounts[x.GetValue()]; ok {
      obj.MargCounts[x.GetValue()] = LogAdd(r, 0.0)
    } else {
      obj.MargCounts[x.GetValue()] = 0.0
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *NonparametricEstimator) updateEstimate() error {
  // recomute bins to match the requested number of bins
  values, counts := obj.computeBins()
  // create new density
  if dist, err := NullDistribution(values); err != nil {
    return err
  } else {
    obj.NonparametricDistribution = dist
  }
  n := NewBareReal(0.0)
  t := NewBareReal(0.0)
  // compute total counts
  for _, c := range counts {
    n.LogAdd(n, ConstReal(c), t)
  }
  for i := 0; i < obj.MargDensity.Dim(); i++ {
    // weight at this position
    w := obj.MargDensity.At(i)
    w.SetValue(counts[i])
    w.Sub(w, n)
    w.Sub(w, ConstReal(math.Log(obj.Delta[i])))
  }
  return nil
}

func (obj *NonparametricEstimator) SetData(x ConstVector, n int) error {
  if err := obj.StdEstimator.SetData(x, n); err != nil {
    return err
  }
  // compute initial histogram
  return obj.Estimate(nil, threadpool.Nil())
}

func (obj *NonparametricEstimator) Estimate(gamma ConstVector, p threadpool.ThreadPool) error {
  x, _ := obj.GetData()
  if err := obj.Initialize(p); err != nil {
    return err
  }
  // update counts
  if gamma == nil {
    for i := 0; i < x.Dim(); i++ {
      if err := obj.NewObservation(x.ConstAt(i), nil, p); err != nil {
        return err
      }
    }
  } else {
    for i := 0; i < x.Dim(); i++ {
      if err := obj.NewObservation(x.ConstAt(i), gamma.ConstAt(i), p); err != nil {
        return err
      }
    }
  }
  if err := obj.updateEstimate(); err != nil {
    return err
  }
  return nil
}

func (obj *NonparametricEstimator) EstimateOnData(x, gamma ConstVector, p threadpool.ThreadPool) error {
  if err := obj.StdEstimator.SetData(x, x.Dim()); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *NonparametricEstimator) GetEstimate() ScalarPdf {
  if obj.MargCounts != nil {
    obj.updateEstimate()
  }
  return obj.NonparametricDistribution
}
