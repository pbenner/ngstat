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

package statistics

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "encoding/json"
import   "io"
import   "io/ioutil"
import   "os"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ConfigDistribution struct {
  Name            string
  Parameters    []float64
  Distributions []ConfigDistribution
}

func NewConfigDistribution(name string, parameters Vector, distributions ...ConfigDistribution) ConfigDistribution {
  var p []float64
  if parameters != nil {
    p = make([]float64, parameters.Dim())
    for i := 0; i < len(p); i++ {
      p[i] = parameters.At(i).GetValue()
    }
  }
  return ConfigDistribution{name, p, distributions}
}

func (config *ConfigDistribution) ReadJson(reader io.Reader) error {
  b, err := ioutil.ReadAll(reader)
  if err != nil {
    return err
  }
  return json.Unmarshal(b, config)
}

func (config *ConfigDistribution) ImportJson(filename string) error {
  f, err := os.Open(filename)
  if err != nil {
    return err
  }
  return config.ReadJson(f)
}

func (config ConfigDistribution) WriteJson(writer io.Writer) error {
  b, err := json.MarshalIndent(config, "", "  ")
  if err != nil {
    return err
  }
  if _, err := writer.Write(b); err != nil {
    return err
  }
  return nil
}

func (config ConfigDistribution) ExportJson(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  return config.WriteJson(f)
}

/* -------------------------------------------------------------------------- */

func ExportDistribution(filename string, distribution BasicDistribution) error {
  return distribution.ExportConfig().ExportJson(filename)
}

/* -------------------------------------------------------------------------- */

func ImportScalarDistributionConfig(config ConfigDistribution, t ScalarType) (ScalarDistribution, error) {
  if distribution := NewScalarDistribution(config.Name); distribution == nil {
    return nil, fmt.Errorf("unknown distribution: %s", config.Name)
  } else {
    if err := distribution.ImportConfig(config, t); err != nil {
      return nil, err
    } else {
      return distribution, err
    }
  }
}

func ImportScalarDistribution(filename string, t ScalarType) (ScalarDistribution, error) {
  config := ConfigDistribution{}

  if err := config.ImportJson(filename); err != nil {
    return nil, err
  }
  return ImportScalarDistributionConfig(config, t)
}

/* -------------------------------------------------------------------------- */

func ImportVectorDistributionConfig(config ConfigDistribution, t ScalarType) (VectorDistribution, error) {
  if distribution := NewVectorDistribution(config.Name); distribution == nil {
    return nil, fmt.Errorf("unknown distribution: %s", config.Name)
  } else {
    if err := distribution.ImportConfig(config, t); err != nil {
      return nil, err
    } else {
      return distribution, err
    }
  }
}

func ImportVectorDistribution(filename string, t ScalarType) (VectorDistribution, error) {
  config := ConfigDistribution{}

  if err := config.ImportJson(filename); err != nil {
    return nil, err
  }
  return ImportVectorDistributionConfig(config, t)
}
