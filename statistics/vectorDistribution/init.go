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

/* -------------------------------------------------------------------------- */

func init() {
  VectorDistributionRegistry["vector:hierarchical hmm distribution"] = new(Hhmm)
  VectorDistributionRegistry["vector:hmm distribution"]              = new(Hmm)
  VectorDistributionRegistry["vector:mixture distribution"]          = new(Mixture)
  VectorDistributionRegistry["vector:normal distribtion"]            = new(NormalDistribution)
  VectorDistributionRegistry["vector:scalar id connector"]           = new(ScalarId)
  VectorDistributionRegistry["vector:scalar iid connector"]          = new(ScalarIid)
  VectorDistributionRegistry["vector:vector id connector"]           = new(VectorId)
  VectorDistributionRegistry["vector:vector iid connector"]          = new(VectorIid)
}
