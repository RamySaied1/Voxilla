#pragma once
#include <math.h>
#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <limits>
using namespace std;
#define uint unsigned int

inline void split(string str, vector<string>& fields) {
    istringstream iss(str);
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(fields));
}

template <class T>
uint moveRelevantFisrt(vector<T>& items, std::function<bool(T&)> isRelevantPred, uint start = 0) {
    uint isOk = start;
    for (uint i = start; i < items.size(); ++i) {
        if (isRelevantPred(items[i])) {
            if (isOk != i) {
                swap(items[isOk], items[i]);
            }
            ++isOk;
        }
    }
    return isOk;
}


inline double scale(double num,pair<double,double> numMinMax,pair<double,double> newMinMax) {
    return num/3;
    // return ((num-numMinMax.first)/( numMinMax.second-numMinMax.first))* (newMinMax.second-newMinMax.first)+newMinMax.first ;
}

class Exception : public exception {
   private:
    string msg;

   public:
    Exception(string msg) : msg(msg) {}
    const char* what() const throw() {
        return msg.c_str();
    }
};

#define printVec(out, cont)       \
    for (const auto& elem : cont) \
        out << elem << " ";       \
    out << endl;

#define print2d(out, matrix)           \
    for (const auto& row : matrix) {   \
        for (const auto& elem : row) { \
            out << elem << " ";        \
        }                              \
        out << endl;                   \
    }

#define read(in, cont)   \
    for (auto& a : cont) \
        in >> a;

#define read2d(in, matrix) \
    for (auto& v : matrix) \
        read(in, v);
