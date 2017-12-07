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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"

import . "github.com/pbenner/ngstat/statistics"
import . "github.com/pbenner/ngstat/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestScalarId(t *testing.T) {

  pi := NewVector(BareRealType, []float64{1, 2})
  tr := NewMatrix(BareRealType, 2, 2, []float64{1,2,3,4})

  d1, _ := NewGammaDistribution(NewReal(1.0), NewReal(2.0), NewReal(3.0))
  d2, _ := NewGammaDistribution(NewReal(2.0), NewReal(3.0), NewReal(4.0))

  hmm1, _ := NewHmm(pi, tr, nil, []ScalarDistribution{d1, d2})

  if err := ExportDistribution("hmm_test.json", hmm1); err != nil {
    t.Error("test failed")
  }
  if hmm2, err := ImportVectorDistribution("hmm_test.json", BareRealType); err != nil {
    t.Error(err)
  } else {
    if Vnorm(VsubV(hmm1.GetParameters(), hmm2.GetParameters())).GetValue() > 1e-10 {
      t.Error("test failed")
    }
  }
}
