/******************************************************************************
 *  Compilation: make
 *  Execution: PatternCount < input.txt
 *  Data files: data1.txt data2.txt data3.txt data4.txt
 *  Implementation of counting how many appearance of a pattern in a text
 *  % PatternCount < data1.txt
 *  2
 *  % PatternCount < data2.txt
 *  3
 *  % PatternCount < data3.txt
 *  4
 *  % PatternCount < data4.txt
 *  28
 *  History:
 *  11-29-15: file created
 ******************************************************************************/
#include <iostream>
#include <string>
#include <cassert>
using std::string;
using std::cin;
using std::cout;
int PatternCount(string text, string pattern) {
  assert(text.length() >= pattern.length());
  int count = 0;
  for (size_t i = 0; i <= (text.length() - pattern.length()); ++i) {
    bool matched = true;
    for (size_t j = 0; j < pattern.length(); ++j) {
      if (text[i + j] != pattern[j]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      ++count;
    }
  }
  return count;
}

int main() {
  string text, pattern;
  cin >> text;
  cin >> pattern;
  cout << PatternCount(text, pattern) << "\n";
  return 0;
}
