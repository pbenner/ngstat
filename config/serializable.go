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

//import "fmt"
import "bytes"
import "io"
import "io/ioutil"
import "os"

/* -------------------------------------------------------------------------- */

type Serializable interface {
  Import(reader io.Reader, args... interface{}) error
  Export(writer io.Writer) error
}

/* -------------------------------------------------------------------------- */

func ImportFile(object Serializable, filename string, args... interface{}) error {
  str, err := ioutil.ReadFile(filename)
  if err != nil {
    return err
  }
  return object.Import(bytes.NewReader(str), args...)
}

func ExportFile(object Serializable, filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  return object.Export(f)
}
