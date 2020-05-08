#include "helpers.hpp"

void split(string str, vector<string>& fields) {
    istringstream iss(str);
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(fields));
}

void strtolower(string& s) { transform(begin(s), end(s), begin(s), ::tolower);}