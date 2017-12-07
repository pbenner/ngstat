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

import . "github.com/pbenner/ngstat/statistics"
import   "github.com/pbenner/ngstat/statistics/generic"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Hmm struct {
  generic.Hmm
  Edist []ScalarDistribution
}

/* -------------------------------------------------------------------------- */

func NewHmm(pi Vector, tr Matrix, stateMap []int, edist []ScalarDistribution) (*Hmm, error) {
  if hmm, err := generic.NewHmm(pi, tr, stateMap, ScalarDistributionSet(edist)); err != nil {
    return nil, err
  } else {
    return &Hmm{*hmm, edist}, nil
  }
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) Clone() *Hmm {
  edist := make([]ScalarDistribution, len(obj.Edist))
  for i := 0; i < len(obj.Edist); i++ {
    edist[i] = obj.Edist[i].CloneScalarDistribution()
  }
  return &Hmm{generic.Hmm{*obj.Hmm.CoreHmm.Clone(), ScalarDistributionSet(edist)}, edist}
}

func (obj *Hmm) CloneVectorDistribution() VectorDistribution {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) Dim() int {
  return -1
}

func (obj *Hmm) LogPdf(r Scalar, x Vector) error {
  return obj.Hmm.LogPdf(r, HmmDataRecord{obj.Edist, x})
}

func (obj *Hmm) Posterior(r Scalar, x Vector, states [][]int) error {
  return obj.Hmm.Posterior(r, HmmDataRecord{obj.Edist, x}, states)
}

func (obj *Hmm) PosteriorMarginals(x Vector) ([]Vector, error) {
  return obj.Hmm.PosteriorMarginals(HmmDataRecord{obj.Edist, x})
}

func (obj *Hmm) Viterbi(x Vector) ([]int, error) {
  return obj.Hmm.Viterbi(HmmDataRecord{obj.Edist, x})
}

/* -------------------------------------------------------------------------- */

func (obj *Hmm) ImportConfig(config ConfigDistribution, t ScalarType) error {

  if err := obj.CoreHmm.ImportConfig(config, t); err != nil {
    return err
  }

  distributions := make([]ScalarDistribution, len(config.Distributions))
  for i := 0; i < len(config.Distributions); i++ {
    if tmp, err := ImportScalarDistributionConfig(config.Distributions[i], t); err != nil {
      return err
    } else {
      distributions[i] = tmp
    }
  }
  obj.Hmm.Edist = ScalarDistributionSet(distributions)
  obj.Edist     = distributions

  return nil
}

func (obj *Hmm) ExportConfig() ConfigDistribution {

  distributions := make([]ConfigDistribution, len(obj.Edist))
  for i := 0; i < len(obj.Edist); i++ {
    distributions[i] = obj.Edist[i].ExportConfig()
  }
  config := obj.CoreHmm.ExportConfig()
  config.Name = "vector hmm distribution"
  config.Distributions = distributions

  return config
}
