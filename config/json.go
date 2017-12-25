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

package config

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "encoding/json"
import   "io"
import   "io/ioutil"

/* -------------------------------------------------------------------------- */

type Config interface {
  Serializable
}

/* -------------------------------------------------------------------------- */

func JsonImport(reader io.Reader, object interface{}) error {

  b, err := ioutil.ReadAll(reader)
  if err != nil {
    return err
  }

  return json.Unmarshal(b, object)
}

func JsonExport(writer io.Writer, object interface{}) error {

  b, err := json.MarshalIndent(object, "", "  ")
  if err != nil {
    return err
  }
  if _, err := writer.Write(b); err != nil {
    return err
  }

  return nil
}
