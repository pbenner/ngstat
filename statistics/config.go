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
import   "reflect"
import   "os"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ConfigDistribution struct {
  Name            string
  Parameters      interface{}
  Distributions []ConfigDistribution
}

func NewConfigDistribution(name string, parameters interface{}, distributions ...ConfigDistribution) ConfigDistribution {
  var p interface{}
  switch parameters := parameters.(type) {
  case []float64:
    p = parameters
  case Vector:
    s := make([]float64, parameters.Dim())
    for i := 0; i < len(s); i++ {
      s[i] = parameters.At(i).GetValue()
    }
    p = s
  default:
    p = parameters
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

func (config ConfigDistribution) getFloat(a interface{}) (float64, bool) {
  switch reflect.TypeOf(a).Kind() {
  case reflect.Float64:
    return reflect.ValueOf(a).Float(), true
  }
  return 0, false
}

func (config ConfigDistribution) getInt(a interface{}) (int, bool) {
  switch reflect.TypeOf(a).Kind() {
  case reflect.Float64:
    return int(reflect.ValueOf(a).Float()), true
  }
  return 0, false
}

func (config ConfigDistribution) getFloats(a interface{}) ([]float64, bool) {
  if a == nil {
    return nil, true
  }
  switch reflect.TypeOf(a).Kind() {
  case reflect.Slice:
    s := reflect.ValueOf(a)
    p := make([]float64, s.Len())
    for i := 0; i < s.Len(); i++ {
      if v, ok := config.getFloat(s.Index(i).Elem().Interface()); !ok {
        return nil, false
      } else {
        p[i] = v
      }
    }
    return p, true
  }
  return nil, false
}

func (config ConfigDistribution) getInts(a interface{}) ([]int, bool) {
  if a == nil {
    return nil, true
  }
  switch reflect.TypeOf(a).Kind() {
  case reflect.Slice:
    s := reflect.ValueOf(a)
    p := make([]int, s.Len())
    for i := 0; i < s.Len(); i++ {
      if v, ok := config.getInt(s.Index(i).Elem().Interface()); !ok {
        return nil, false
      } else {
        p[i] = v
      }
    }
    return p, true
  }
  return nil, false
}

func (config ConfigDistribution) GetParametersAsFloats() ([]float64, bool) {
  return config.getFloats(config.Parameters)
}

func (config ConfigDistribution) GetParametersAsVector(t ScalarType) (Vector, bool) {
  if v, ok := config.getFloats(config.Parameters); !ok {
    return nil, false
  } else {
    return NewVector(t, v), true
  }
}

func (config ConfigDistribution) GetParametersAsMatrix(t ScalarType, n, m int) (Matrix, bool) {
  if v, ok := config.getFloats(config.Parameters); !ok {
    return nil, false
  } else {
    return NewMatrix(t, n, m, v), true
  }
}

func (config ConfigDistribution) GetNamedParameter(name string) (interface{}, bool) {
  switch reflect.TypeOf(config.Parameters).Kind() {
  case reflect.Map:
    s := reflect.ValueOf(config.Parameters)
    r := s.MapIndex(reflect.ValueOf(name))
    if r.IsValid() {
      return r.Interface(), true
    }
  }
  return 0, false
}

func (config ConfigDistribution) GetNamedParametersAsFloats(name string) ([]float64, bool) {
  if p, ok := config.GetNamedParameter(name); ok {
    return config.getFloats(p)
  }
  return nil, false
}

func (config ConfigDistribution) GetNamedParametersAsInts(name string) ([]int, bool) {
  if p, ok := config.GetNamedParameter(name); ok {
    return config.getInts(p)
  }
  return nil, false
}

func (config ConfigDistribution) GetNamedParameterAsFloat(name string) (float64, bool) {
  if p, ok := config.GetNamedParameter(name); ok {
    return config.getFloat(p)
  }
  return 0, false
}

func (config ConfigDistribution) GetNamedParameterAsInt(name string) (int, bool) {
  if p, ok := config.GetNamedParameter(name); ok {
    return config.getInt(p)
  }
  return 0, false
}

func (config ConfigDistribution) GetNamedParametersAsVector(name string, t ScalarType) (Vector, bool) {
  if v, ok := config.GetNamedParametersAsFloats(name); !ok {
    return nil, false
  } else {
    return NewVector(t, v), true
  }
}

func (config ConfigDistribution) GetNamedParametersAsMatrix(name string, t ScalarType, n, m int) (Matrix, bool) {
  if v, ok := config.GetNamedParametersAsFloats(name); !ok {
    return nil, false
  } else {
    return NewMatrix(t, n, m, v), true
  }
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

/* -------------------------------------------------------------------------- */

func ImportMatrixDistributionConfig(config ConfigDistribution, t ScalarType) (MatrixDistribution, error) {
  if distribution := NewMatrixDistribution(config.Name); distribution == nil {
    return nil, fmt.Errorf("unknown distribution: %s", config.Name)
  } else {
    if err := distribution.ImportConfig(config, t); err != nil {
      return nil, err
    } else {
      return distribution, err
    }
  }
}

func ImportMatrixDistribution(filename string, t ScalarType) (MatrixDistribution, error) {
  config := ConfigDistribution{}

  if err := config.ImportJson(filename); err != nil {
    return nil, err
  }
  return ImportMatrixDistributionConfig(config, t)
}
