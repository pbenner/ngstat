/* Copyright (C) 2016, 2017 Philipp Benner
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

package track

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/ngstat/io"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func MaskRegionsOnTrack(config SessionConfig, track Track, r GRanges) error {
  PrintStderr(config, 1, "Masking regions... ")
  for i := 0; i < r.Length(); i++ {
    if seq, err := track.GetSlice(r.Row(i)); err == nil {
      for j := 0; j < len(seq); j++ {
        seq[j] = math.NaN()
      }
    } else {
      PrintStderr(config, 1, "failed\n")
      return fmt.Errorf("error while trying to exclude region: %v", err)
    }
  }
  PrintStderr(config, 1, "done\n")
  return nil
}
