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

package vectorEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "os"
import   "math"
import   "testing"

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/scalarDistribution"
import   "github.com/pbenner/ngstat/statistics/vectorDistribution"
import   "github.com/pbenner/ngstat/statistics/scalarEstimator"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"
import   "github.com/pbenner/autodiff/algorithm/bfgs"
import   "github.com/pbenner/autodiff/algorithm/newton"

import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func TestHmm1(t *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  pi := NewVector(RealType, []float64{0.6, 0.4})
  tr := NewMatrix(RealType, 2, 2,
    []float64{0.7, 0.3, 0.4, 0.6})

  e1, _ := scalarEstimator.NewCategoricalEstimator(
    NewVector(RealType, []float64{0.1, 0.9}))
  e2, _ := scalarEstimator.NewCategoricalEstimator(
    NewVector(RealType, []float64{0.7, 0.3}))

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, nil)
  if err != nil {
    t.Error(err)
  }
  // test Baum-Welch algorithm
  //////////////////////////////////////////////////////////////////////////////
  if estimator, err := NewHmmEstimator(hmm, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
    t.Error(err)
  } else {
    x := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})

    if err := estimator.EstimateOnData([]Vector{x}, nil, ThreadPool{}); err != nil {
      t.Error(err)
    } else {
      hmm1 := hmm
      hmm2 := estimator.GetEstimate()

      p1 := NullReal(); hmm1.LogPdf(p1, x)
      p2 := NullReal(); hmm2.LogPdf(p2, x)

      if p1.Greater(p2) {
        t.Errorf("Baum-Welch test failed")
      }
      if math.Abs(p2.GetValue() - -4.493268e+00) > 1e-4 {
        t.Errorf("Baum-Welch test failed")
      }
    }
  }
  // test Baum-Welch algorithm with conditioning
  //////////////////////////////////////////////////////////////////////////////
  {
    hmm := hmm.Clone()
    hmm.SetStartStates([]int{0})
    hmm.SetFinalStates([]int{0})

    if estimator, err := NewHmmEstimator(hmm, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
      t.Error(err)
    } else {
      x  := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})

      if err := estimator.EstimateOnData([]Vector{x}, nil, ThreadPool{}); err != nil {
        t.Error(err)
      } else {
        hmm1 := hmm
        hmm2 := estimator.GetEstimate()

        p1 := NullReal(); hmm1.LogPdf(p1, x)
        p2 := NullReal(); hmm2.LogPdf(p2, x)
        if p1.Greater(p2) {
          t.Errorf("Baum-Welch test failed")
        }
        if math.Abs(p2.GetValue() - -5.834855e+00) > 1e-4 {
          t.Errorf("Baum-Welch test failed")
        }
      }
    }
  }
}

func TestHmm2(t *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  tr := NewMatrix(RealType, 2, 2,
    []float64{0.7, 0.3, 0.4, 0.6})

  c1, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.1, 0.9}))
  c2, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.7, 0.3}))
  edist := []ScalarDistribution{c1, c2}

  pi := NewVector(RealType, []float64{0.6, 0.4})

  x  := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})
  r  := NewReal(0.0)

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
  if err != nil {
    t.Error(err)
  }
  hmm.SetStartStates([]int{0})
  hmm.SetFinalStates([]int{0})

  penalty := func(p1, p2, c Scalar) Scalar {
    r := NewReal(0.0)
    r.Add(p1, p2)
    r.Sub(r, NewReal(1.0))
    r.Pow(r, NewReal(2.0))
    r.Mul(r, c)
    return r
  }
  objective_template := func(variables Vector, c Scalar) (Scalar, error) {
    // create a new initial normal distribution
    pi := NullVector(RealType, 2)
    tr := NullMatrix(RealType, 2, 2)
    // copy the variables
    pi.At(0).SetValue(1.0)
    pi.At(1).SetValue(1.0)
    tr.At(0, 0).Exp(variables.At(0))
    tr.At(0, 1).Exp(variables.At(1))
    tr.At(1, 0).Exp(variables.At(2))
    tr.At(1, 1).Exp(variables.At(3))
    // construct new Hmm
    hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
    hmm.SetStartStates([]int{0})
    hmm.SetFinalStates([]int{0})
    // compute objective function
    result := NewScalar(RealType, 0.0)
    // density function
    hmm.LogPdf(r, x)
    result.Add(result, r)
    result.Neg(result)
    // penalty function
    result.Add(result, penalty(pi.At(0),pi.At(1), c))
    result.Add(result, penalty(tr.At(0, 0),tr.At(0, 1), c))
    result.Add(result, penalty(tr.At(1, 0),tr.At(1, 1), c))
    return result, nil
  }
  // hook_bfgs := func(variables, gradient Vector, s Scalar) bool {
  //   fmt.Println("variables:", variables)
  //   fmt.Println("gradient :", gradient)
  //   fmt.Println("y        :", s)
  //   fmt.Println("")
  //   return false
  // }
  // initial value
  vn := hmm.GetParameters()
  vn  = vn.Slice(2,vn.Dim())
  // initial penalty strength
  c  := NewReal(2.0)
  // run rprop
  for i := 0; i < 20; i++ {
    objective := func(variables Vector) (Scalar, error) {
      return objective_template(variables, c)
    }
    vn, _ = bfgs.Run(objective, vn,
      //bfgs.Hook{hook_bfgs},
      bfgs.Epsilon{1e-8})
    // increase penalty strength
    c.Mul(c, NewReal(2.0))
  }
  // check result
  if math.Abs(Exp(vn.At(0)).GetValue() - 8.257028e-01) > 1e-3 ||
     math.Abs(Exp(vn.At(1)).GetValue() - 1.743001e-01) > 1e-3 ||
     math.Abs(Exp(vn.At(2)).GetValue() - 3.597875e-01) > 1e-3 ||
     math.Abs(Exp(vn.At(3)).GetValue() - 6.402134e-01) > 1e-3 {
    t.Error("Hmm test failed!")
  }
}

func TestHmm3(t *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  tr := NewMatrix(RealType, 2, 2,
    []float64{5,1,1,10})

  c1, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.1, 0.9}))
  c2, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.7, 0.3}))
  edist := []ScalarDistribution{c1, c2}

  pi := NewVector(RealType, []float64{0.6, 0.4})

  x  := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})
  r  := NewReal(0.0)

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
  if err != nil {
    t.Error(err)
  }

  constraint := func(p1, p2, lambda Scalar) Scalar {
    r := NewReal(0.0)
    r.Add(p1, p2)
    r.Sub(r, NewReal(1.0))
    r.Mul(r, lambda)
    return r
  }
  objective := func(variables Vector) (Scalar, error) {
    // create a new initial normal distribution
    tr := NullMatrix(RealType, 2, 2)
    // copy the variables
    tr.At(0, 0).Exp(variables.At(0))
    tr.At(0, 1).Exp(variables.At(1))
    tr.At(1, 0).Exp(variables.At(2))
    tr.At(1, 1).Exp(variables.At(3))
    // lambda parameters of the Lagrangian
    lambda := variables.Slice(4,6)
    // construct new Hmm
    hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
    // compute objective function
    result := NewScalar(RealType, 0.0)
    // density function
    hmm.LogPdf(r, x)
    result.Add(result, r)
    result.Neg(result)
    // constraints
    result.Add(result, constraint(tr.At(0, 0), tr.At(0, 1), lambda.At(0)))
    result.Add(result, constraint(tr.At(1, 0), tr.At(1, 1), lambda.At(1)))
    return result, nil
  }
  // hook_newton := func(x Vector, hessian Matrix, gradient Vector) bool {
  //   fmt.Println("hessian :", hessian)
  //   fmt.Println("gradient:", gradient)
  //   fmt.Println("x       :", x)
  //   fmt.Println("")
  //   return false
  // }
  // initial value
  vn := hmm.GetParameters()
  // drop pi and parameters from the emission distributions
  vn  = vn.Slice(2,6)
  // append Lagriangian lambda parameters
  vn  = vn.AppendScalar(NewReal(1.0), NewReal(1.0))
  // find critical points of the Lagrangian
  vn, err = newton.RunCrit(objective, vn,
//    newton.HookCrit{hook_newton},
    newton.Epsilon{1e-4})
  if err != nil {
    t.Error(err)
  } else {
    // check result
    if math.Abs(Exp(vn.At(0)).GetValue() - 8.230221e-01) > 1e-4 ||
       math.Abs(Exp(vn.At(1)).GetValue() - 1.769779e-01) > 1e-4 ||
       math.Abs(Exp(vn.At(2)).GetValue() - 7.975104e-09) > 1e-4 ||
       math.Abs(Exp(vn.At(3)).GetValue() - 1.000000e+00) > 1e-4 {
      t.Error("Hmm test failed!")
    }
  }
}

func TestHmm4(t *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  pi := NewVector(RealType, []float64{0.6, 0.4})

  tr := NewMatrix(RealType, 2, 2,
    []float64{0.7, 0.3, 0.4, 0.6})

  c1, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.1, 0.9}))
  c2, _ := scalarDistribution.NewCategoricalDistribution(
    NewVector(RealType, []float64{0.7, 0.3}))
  edist := []ScalarDistribution{c1, c2}

  x  := NewVector(RealType, []float64{1,1,1,1,1,1,0,0,1,0})
  r  := NewReal(0.0)

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
  if err != nil {
    t.Error(err)
  }
  hmm.SetStartStates([]int{0})
  hmm.SetFinalStates([]int{0})

  constraint := func(p1, p2, lambda Scalar) Scalar {
    r := NewReal(0.0)
    r.Add(p1, p2)
    r.Sub(r, NewReal(1.0))
    r.Mul(r, lambda)
    return r
  }
  objective := func(variables Vector) (Scalar, error) {
    // create a new initial normal distribution
    tr := NullMatrix(RealType, 2, 2)
    // copy the variables
    tr.At(0, 0).Exp(variables.At(0))
    tr.At(0, 1).Exp(variables.At(1))
    tr.At(1, 0).Exp(variables.At(2))
    tr.At(1, 1).Exp(variables.At(3))
    // lambda parameters of the Lagrangian
    lambda := variables.Slice(4,6)
    // construct new Hmm
    hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
    hmm.SetStartStates([]int{0})
    hmm.SetFinalStates([]int{0})
    // compute objective function
    result := NewScalar(RealType, 0.0)
    // density function
    hmm.LogPdf(r, x)
    result.Add(result, r)
    result.Neg(result)
    // constraints
    result.Add(result, constraint(tr.At(0, 0), tr.At(0, 1), lambda.At(0)))
    result.Add(result, constraint(tr.At(1, 0), tr.At(1, 1), lambda.At(1)))
    return result, nil
  }
  // hook_newton := func(x Vector, hessian Matrix, gradient Vector) bool {
  //   fmt.Println("hessian :", hessian)
  //   fmt.Println("gradient:", gradient)
  //   fmt.Println("x       :", x)
  //   fmt.Println("")
  //   return false
  // }
  // initial value
  vn := hmm.GetParameters()
  // drop pi and parameters from the emission distributions
  vn  = vn.Slice(2,6)
  // append Lagriangian lambda parameters
  vn  = vn.AppendScalar(NewReal(1.0), NewReal(1.0))
  // run rprop
  vn, err = newton.RunCrit(objective, vn,
    //newton.HookCrit{hook_newton},
    newton.Epsilon{1e-10})
  if err != nil {
    t.Error(err)
  } else {
    // check result
    if math.Abs(Exp(vn.At(0)).GetValue() - 8.257028e-01) > 1e-3 ||
       math.Abs(Exp(vn.At(1)).GetValue() - 1.743001e-01) > 1e-3 ||
       math.Abs(Exp(vn.At(2)).GetValue() - 3.597875e-01) > 1e-3 ||
       math.Abs(Exp(vn.At(3)).GetValue() - 6.402134e-01) > 1e-3 {
      t.Error("Hmm test failed!")
    }
  }
}

func TestHmm5(t *testing.T) {

  pi := NewVector(RealType, []float64{0.6, 0.4})

  var tr Matrix = NewMatrix(RealType, 2, 2,
    []float64{0.7, 0.3, 0.4, 0.6})

  c1, _ := scalarDistribution.NewGammaDistribution(NewReal( 0.5), NewReal(2.0), NewReal(0))
  c2, _ := scalarDistribution.NewGammaDistribution(NewReal(10.0), NewReal(2.0), NewReal(0))

  e1, _ := scalarEstimator.NewNumericEstimator(c1)
  e2, _ := scalarEstimator.NewNumericEstimator(c2)

  e1.Epsilon = 1e-7
  e2.Epsilon = 1e-7

  x  := []Vector{
    NewVector(RealType, []float64{0.23092451, 0.23092451, 0.23092451, 5.975650, 5.975650, 5.975650}),
    NewVector(RealType, []float64{1.15626248, 1.15626248, 1.15626248, 3.074001, 3.074001, 3.074001}),
    NewVector(RealType, []float64{0.39937995, 0.39937995, 0.39937995, 3.806467, 3.806467, 3.806467}),
    NewVector(RealType, []float64{0.51252240, 0.51252240, 0.51252240, 6.654319, 6.654319, 6.654319}),
    NewVector(RealType, []float64{2.35671304, 2.35671304, 2.35671304, 2.904598, 2.904598, 2.904598}),
    NewVector(RealType, []float64{0.18067285, 0.18067285, 0.18067285, 2.895080, 2.895080, 2.895080}),
    NewVector(RealType, []float64{0.06068149, 0.06068149, 0.06068149, 3.088718, 3.088718, 3.088718}),
    NewVector(RealType, []float64{1.71700325, 1.71700325, 1.71700325, 4.068132, 4.068132, 4.068132}),
    NewVector(RealType, []float64{0.06229591, 0.06229591, 0.06229591, 4.466460, 4.466460, 4.466460}),
    NewVector(RealType, []float64{0.43543498, 0.43543498, 0.43543498, 6.193897, 6.193897, 6.193897}) }

  hmm, err := vectorDistribution.NewHmm(pi, tr, nil, nil)
  if err != nil {
    t.Error(err); return
  }
  estimator, err := NewHmmEstimator(hmm, []ScalarEstimator{e1, e2}, 1e-10, -1)
  if err != nil {
    t.Error(err); return
  }
  if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
    t.Error(err); return
  }
  hmm = estimator.GetEstimate().(*vectorDistribution.Hmm)

  // correct values
  qi := NewVector(RealType, []float64{7.249908e-01, 2.750092e-01})
  sr := NewMatrix(RealType, 2, 2, []float64{
    6.592349e-01, 3.407651e-01,
    0.000000e+00, 1.000000e+00 })

  pi = hmm.Pi; pi.MapSet(Exp)
  tr = hmm.Tr; tr.MapSet(Exp)

  if Vnorm(VsubV(pi, qi)).GetValue() > 1e-3 {
    t.Error("Hmm test failed!")
  }
  if Mnorm(MsubM(tr, sr)).GetValue() > 1e-4 {
    t.Error("Hmm test failed!")
  }
  if math.Abs(hmm.Edist[0].GetParameters().At(0).GetValue() - 1.792786e+00) > 1e-4 {
    t.Error("Hmm test failed!")
  }
  if math.Abs(hmm.Edist[0].GetParameters().At(1).GetValue() - 6.371870e+00) > 1e-4 {
    t.Error("Hmm test failed!")
  }
  if math.Abs(hmm.Edist[1].GetParameters().At(0).GetValue() - 4.855799e+00) > 1e-4 {
    t.Error("Hmm test failed!")
  }
  if math.Abs(hmm.Edist[1].GetParameters().At(1).GetValue() - 1.299225e+00) > 1e-4 {
    t.Error("Hmm test failed!")
  }
}
