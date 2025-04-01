# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate accuracy of JetStream online serving."""
import re

from sympy import sympify
from sympy.core.sympify import SympifyError
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.latex import parse_latex


def replace_space_answers(text):
  return text.replace(" ", "")


def extract_numbers(string):
  # Extract numbers from the sentence, i.e. The length is 0.3 -> 0.3
  sentence = string.split(" ")
  if len(sentence) > 1:
    for word in sentence:
      if re.match(r"[+\-*/\d]+", word):
        return word
  return string


def fix_sqrt(string):
  # Fix sqrtxx -> sqrt{xx}
  string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
  string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", string)
  return string


def fix_a_slash_b(string):
  # Replace a/b to frac{a}{b}
  if len(string.split("/")) != 2:
    return string
  a = string.split("/")[0]
  b = string.split("/")[1]
  try:
    if "sqrt" not in a:
      a = int(a)
    if "sqrt" not in b:
      b = int(b)
    assert string == f"{a}/{b}"
    new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
    return new_string
  except ValueError:
    return string


def fix_tan(string):
  string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
  string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", string)
  return string


def fix_fracs(string):
  # Standardize frac to the format of frac{a}{b}
  string = string.replace("tfrac", "frac")
  string = string.replace("dfrac", "frac")
  string = string.replace("cfrac", "frac")
  substrs = string.split("\\frac")
  new_str = substrs[0]
  if len(substrs) > 1:
    substrs = substrs[1:]
    for substr in substrs:
      new_str += "\\frac"

      def _replace_match(match):
        groups = match.groups()
        return "{" + "}{".join(groups[:-1]) + "}" + groups[-1]

      patterns = [
          r"\{(\S+)\}\{(\S+)\}(.*)",  # match frac{X}{Y}
          r"(\S+)\{(\S+)\}(.*)",  # match fracX{Y}
          r"\{(\S+)\}(\S)(.*)",  # match frac{X}Y
          r"(\S)(\S)(.*)",  # match fracXY
      ]

      reformat_str = ""
      for pattern in patterns:
        match = re.match(pattern, substr)
        if match:
          reformat_str = _replace_match(match)
          break
      if reformat_str:
        new_str += reformat_str
      else:
        return string

  string = new_str
  return string


def fix_base_term(string):
  # Fix base_term, i.e. 100_8 -> 100_{8}
  match = re.match(r"(\d+)_(\d+)", string)
  if match:
    return match.group(1) + "_{" + match.group(2) + "}"
  else:
    return string


def expand_pm(string):
  # Expand pm expression to + and -
  match = re.match(r"([^,]+)\\pm([^,]+)", string)
  if match:
    plus_term = match.group(1) + "+" + match.group(2)
    mins_term = match.group(1) + "-" + match.group(2)
    return set({plus_term, mins_term})
  else:
    return set({string})


def parse_set(string):
  # Parse set expression to python list
  if isinstance(string, list):
    return string
  match = re.match(r"\\\{(.*)\\\}$", string.strip())
  if match:
    elements_string = match.group(1).strip()
    elements = [elem.strip() for elem in elements_string.split(",")]
    return elements
  else:
    return [string]


def remove_commas_from_numbers(string):
  # Remove commas within large numbers i.e., 1,234 -> 1234
  number_pattern = r"(?<!\()(\d{1,3}(?:,\d{3})+)(?!\))"
  return re.sub(
      number_pattern, lambda match: match.group(0).replace(",", ""), string
  )


def convert_leading_zero(string):
  # Add "0" if "." is the start of the string
  string = string.replace(" .", " 0.")
  string = string.replace("{.", "{0.")
  return string


def trim_general_unit(string):
  # Remove unit: miles, dollars if after is not none
  new_string = re.sub(r"\\text{.*?}$", "", string).strip()
  if new_string != "" and new_string != string:
    print(f"Warning: unit not removed: '{string}' -> '{new_string}'")
    string = new_string
  string = re.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
  string = re.sub(r"miles(\^(2|3))?", "", string).strip()
  string = re.sub(r"p\.m\.$", "", string).strip()
  string = re.sub(r"(\d)\s*t$", r"\1", string).strip()

  # Remove degree:
  string = re.sub(r"\^\\circ", "", string)

  return string


def postprocess_math(text):

  text = trim_latex_cmd(text)

  # Remove the units
  text = trim_general_unit(text)

  # Extract the answer in \text command
  match = re.search(r"\\text\{([^}]*)\}", text)
  if match:
    text = match.group(1)

  # Remove space
  text = replace_space_answers(text)

  # Standardize sqrt format as sqrt{}/{}
  text = fix_sqrt(text)

  # Standardize frac format
  text = fix_fracs(text)

  # Convert {}/{} to the format of frac {} {}
  text = fix_a_slash_b(text)

  # Handle corner cases
  text = special_handling(text)

  return text


def trim_latex_cmd(text):
  text = text.replace("\\left", "")
  text = text.replace("\\right", "")
  text = re.sub(r"\\mbox{.*?}", "", text)

  return text


def latex_matrix_to_list(latex_string):
  # Convert LaTex Matrix expression into 2d Array
  if latex_string:
    try:
      # Extract matrix content
      matrix_content = re.search(
          r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", latex_string, re.DOTALL
      )
      if matrix_content:
        rows = matrix_content.group(1).strip().split("\\\\")
      else:
        return None
      matrix_list = []
      for row in rows:
        cols = row.split("&")
        matrix_list.append([col.strip() for col in cols])
      return matrix_list
    except ValueError as e:
      print(f"Error parsing LaTeX matrix: {e}")
      return None

  return None


def sympify_set(text_set):
  sympified_set = set()
  for element in text_set:
    try:
      if "frac" in element:
        try:
          element = parse_latex(element)
        except LaTeXParsingError:
          pass
      sympified_set.add(sympify(element).evalf())
    except (SympifyError, AttributeError, TypeError):
      sympified_set.add(element)
  return sympified_set


def sympify_matrix(matrix):
  if matrix:
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    for row in range(num_rows):
      for col in range(num_cols):
        matrix[row][col] = replace_space_answers(matrix[row][col])
        matrix[row][col] = fix_sqrt(matrix[row][col])
        matrix[row][col] = fix_fracs(matrix[row][col])
        matrix[row][col] = fix_a_slash_b(matrix[row][col])
        matrix[row][col] = parse_latex(matrix[row][col])
        matrix[row][col] = sympify(matrix[row][col])

  return matrix


def special_handling(string):

  # Remove \$ symbol for LaTex
  string = re.sub(r"\\\$", "", string)

  # Remove \th for Latex
  string = re.sub(r"\^\{\\mathrm\{th\}\}", "", string)

  # Remove , from numbers
  string = remove_commas_from_numbers(string)

  # Fix base term
  string = fix_base_term(string)
  # Expand pm or mp command
  string_set = expand_pm(string)

  # Convert LaTex format to sympy format
  try:
    string_set = parse_latex(string_set)
  except (LaTeXParsingError, AttributeError):
    pass

  return string_set


def post_processing_math_ans(answers):
  output_set = set()
  for ans in parse_set(answers):
    matrix = latex_matrix_to_list(ans)
    if matrix:
      output_set.add(str(sympify_matrix(matrix)))
    else:
      ans = postprocess_math(ans)
      output_set.update(ans)
  return output_set
