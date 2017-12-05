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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "reflect"

import . "github.com/pbenner/ngstat/statistics"

/* -------------------------------------------------------------------------- */

var ScalarDistributionRegistry map[string]ScalarDistribution

func init() {
  ScalarDistributionRegistry = make(map[string]ScalarDistribution)
  ScalarDistributionRegistry["beta distribution"]               = new(BetaDistribution)
  ScalarDistributionRegistry["binomial distribution"]           = new(BinomialDistribution)
  ScalarDistributionRegistry["negative binomial distribution"]  = new(NegativeBinomialDistribution)
  ScalarDistributionRegistry["categorical distribution"]        = new(CategoricalDistribution)
  ScalarDistributionRegistry["gamma distribution"]              = new(GammaDistribution)
  ScalarDistributionRegistry["generalized gamma distribution"]  = new(GeneralizedGammaDistribution)
  ScalarDistributionRegistry["nonparametric distribution"]      = new(NonparametricDistribution)
  ScalarDistributionRegistry["gev distribution"]                = new(GevDistribution)
  ScalarDistributionRegistry["exponential distribution"]        = new(ExponentialDistribution)
  ScalarDistributionRegistry["pareto distribution"]             = new(ParetoDistribution)
  ScalarDistributionRegistry["generalized pareto distribution"] = new(GParetoDistribution)
  ScalarDistributionRegistry["power law distribution"]          = new(PowerLawDistribution)
  ScalarDistributionRegistry["log-cauchy distribution"]         = new(LogCauchyDistribution)
  ScalarDistributionRegistry["dirac distribution"]              = new(DiracDistribution)

}

func NewScalarDistribution(name string) ScalarDistribution {
  if x, ok := ScalarDistributionRegistry[name]; ok {
    return reflect.New(reflect.TypeOf(x).Elem()).Interface().(ScalarDistribution)
  } else {
    return nil
  }
}
