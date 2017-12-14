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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/ngstat/statistics"

/* -------------------------------------------------------------------------- */

func init() {
  ScalarDistributionRegistry["scalar:beta distribution"]               = new(BetaDistribution)
  ScalarDistributionRegistry["scalar:binomial distribution"]           = new(BinomialDistribution)
  ScalarDistributionRegistry["scalar:negative binomial distribution"]  = new(NegativeBinomialDistribution)
  ScalarDistributionRegistry["scalar:categorical distribution"]        = new(CategoricalDistribution)
  ScalarDistributionRegistry["scalar:gamma distribution"]              = new(GammaDistribution)
  ScalarDistributionRegistry["scalar:generalized gamma distribution"]  = new(GeneralizedGammaDistribution)
  ScalarDistributionRegistry["scalar:nonparametric distribution"]      = new(NonparametricDistribution)
  ScalarDistributionRegistry["scalar:normal distribution"]             = new(NormalDistribution)
  ScalarDistributionRegistry["scalar:log-normal distribution"]         = new(LogNormalDistribution)
  ScalarDistributionRegistry["scalar:gev distribution"]                = new(GevDistribution)
  ScalarDistributionRegistry["scalar:exponential distribution"]        = new(ExponentialDistribution)
  ScalarDistributionRegistry["scalar:pareto distribution"]             = new(ParetoDistribution)
  ScalarDistributionRegistry["scalar:generalized pareto distribution"] = new(GParetoDistribution)
  ScalarDistributionRegistry["scalar:power law distribution"]          = new(PowerLawDistribution)
  ScalarDistributionRegistry["scalar:log-cauchy distribution"]         = new(LogCauchyDistribution)
  ScalarDistributionRegistry["scalar:dirac distribution"]              = new(DiracDistribution)
}
