/* Copyright (C) 2017 Philipp Benner
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

package scalarEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
//import   "os"
import   "testing"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/generic"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func Test1(t *testing.T) {

  weights := NewVector(BareRealType, []float64{1.0, 2.0})

  mixture, _ := scalarDistribution.NewMixture(weights, nil)

  e1, _ := NewNormalEstimator(0, 2, 0)
  e2, _ := NewNormalEstimator(3, 2, 0)

  likelihood := 0.0
  hook_f := func(mixture generic.BasicMixture, i int, likelihood_, epsilon float64) {
    likelihood = likelihood_
  }

  estimator, err := NewMixtureEstimator(mixture, []ScalarEstimator{e1, e2}, 1e-8, -1, generic.EmHook{hook_f}); if err != nil {
    t.Error(err); return
  }

  x := NewVector(BareRealType, []float64{
    // > rnorm(100, -2, 1)
    -1.12431376, -2.20795920, -1.92180583, -3.83095994, -0.29385167, -1.48853995,
    -4.19290038, -2.19522679, -1.60687622, -3.04262028, -1.35188740, -3.41378977,
    -1.96810219, -1.51452271, -0.62235783, -2.43573938, -2.03582642, -0.07893768,
    -1.83944881, -1.92728205, -1.50059327, -3.14082240, -1.55467447, -2.18027008,
    -3.01752686,  0.48156923, -1.12768807, -2.76360235, -2.65887114, -0.42811274,
    -0.05941949, -3.45760518, -3.62681707, -1.84123854, -1.03036799, -2.49029127,
    -2.94122179, -3.05063883, -0.72819430, -2.96888308, -1.96512102, -1.41604192,
    -1.04308536, -1.85277544, -1.66095761, -2.67491902, -1.54995242, -1.20348112,
    -1.46677152, -2.31217821, -0.52540743, -1.05310707, -4.02416985, -0.23648571,
    -2.51341021, -2.12569988, -4.26158826, -3.41528525, -0.87935292, -2.28957023,
    -1.71899483, -3.51891741, -1.74804924, -2.39061249, -1.48617454, -2.19115759,
    -2.66131490, -2.88021916, -1.77827777, -2.11446584, -1.78444776, -3.27350950,
    -1.79337664, -2.37602476, -3.02796325, -1.26123942, -1.91542623, -2.66857833,
    -1.67410892, -2.94117885, -1.40776103, -0.30426579, -1.38215717, -3.78800171,
    -1.50254451, -2.31069473, -1.68790441, -3.06557311,  0.22761985, -2.65347861,
    -1.86817988, -3.63464871, -1.77518252, -2.59174815, -1.76654045, -0.62347757,
    -3.15778292, -1.07791058, -3.95905908, -2.30306063,
    // > rnorm(100, 2, 1)
     0.58553676,  2.32643787,  0.06887515,  3.57135491,  0.66510976,  2.73711501,
     2.73393575, -0.43275045,  1.18719999,  2.36580560,  2.42780598,  0.45223546,
     0.43574499, -1.24248845,  2.99690483,  0.19019933,  1.48353119,  3.70310066,
     2.67731421,  2.39872148,  3.10838213,  3.02935885,  1.95200227,  0.58949131,
     2.30411652,  1.37612188,  1.01619068,  2.39607721,  1.47663253,  2.13606516,
     1.91437488,  1.18780933,  1.34935317,  2.50250145,  1.75660250,  2.66791023,
     1.27169293,  3.00693522,  1.57234459,  1.44393803,  0.99462070,  3.03269228,
     2.85227513,  2.67160517,  1.63476206,  1.74635511,  0.49677751,  3.51552380,
     1.01361725,  3.72707398,  0.97465116,  0.93906827,  1.68913812,  1.92000453,
     1.32573910,  2.06215109,  1.09815945,  1.19909364,  0.16161479,  1.30767343,
     3.07222179,  1.78362170,  2.22070573,  2.24063169,  3.54774950,  0.38119226,
     1.75711504,  4.59615102,  2.36242738,  1.67403831,  1.51014775,  2.42360094,
     2.83448637,  0.21421829,  1.71037214,  3.18370094,  2.08497781,  2.27872132,
     3.93961299,  2.32686623,  1.75698913,  2.93323066,  1.32865698, -0.36090046,
    -0.16809516,  1.96128465,  1.26031262,  1.92023726,  1.54853494,  1.47283965,
     1.11630221,  2.40132588,  2.55262575,  2.34633999,  2.32867420,  1.96352877,
     1.38095964,  2.81542013,  1.60268309,  0.28949776, })

  if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
    t.Error(err)
  }
  if math.Abs(likelihood - -411.8011450422043) > 1e-6 {
    t.Error("test failed")
  }
}
