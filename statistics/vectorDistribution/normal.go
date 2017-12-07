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

package vectorDistribution

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/ngstat/statistics"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/distribution"

/* -------------------------------------------------------------------------- */

type NormalDistribution struct {
  distribution.NormalDistribution
}

/* -------------------------------------------------------------------------- */

func NewNormalDistribution(mu Vector, sigma Matrix) (*NormalDistribution, error) {
  if dist, err := distribution.NewNormalDistribution(mu, sigma); err != nil {
    return nil, err
  } else {
    return &NormalDistribution{*dist}, err
  }
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) Clone() *NormalDistribution {
  return &NormalDistribution{*obj.NormalDistribution.Clone()}
}

func (obj *NormalDistribution) CloneVectorDistribution() VectorDistribution {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *NormalDistribution) ImportConfig(config ConfigDistribution, t ScalarType) error {

  n, ok := config.GetNamedParameterAsInt("N"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  mu, ok := config.GetNamedParametersAsVector("Mu", t); if !ok {
    return fmt.Errorf("invalid config file")
  }
  sigma, ok := config.GetNamedParametersAsMatrix("Sigma", t, n, n); if !ok {
    return fmt.Errorf("invalid config file")
  }

  if tmp, err := NewNormalDistribution(mu, sigma); err != nil {
    return err
  } else {
    *obj = *tmp
  }
  return nil
}

func (obj *NormalDistribution) ExportConfig() ConfigDistribution {

  n := obj.Dim()

  config := struct{
    Mu    []float64
    Sigma []float64
    N       int }{}
  config.Mu    = make([]float64, n)
  config.Sigma = make([]float64, n*n)
  config.N     = n

  for i := 0; i < n; i++ {
    config.Mu[i] = obj.Mu.At(i).GetValue()
    for j := 0; j < n; j++ {
      config.Sigma[i*n+j] = obj.Sigma.At(i,j).GetValue()
    }
  }
  return NewConfigDistribution("multivariate normal distribtion", config)
}
