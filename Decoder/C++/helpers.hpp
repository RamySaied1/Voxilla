#pragma once
#include <math.h>
#include <fst/fstlib.h>
#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>
using namespace std;
#define uint unsigned int

void split(string str, vector<string>& fields);

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

void strtolower(string& s);
void writeFst(fst::StdVectorFst& fst,string filename);

class Exception : public exception {
   private:
    string msg;

   public:
    Exception(string msg) : msg(msg) {}
    const char* what() const throw() {
        return msg.c_str();
    }
};

struct HashPair {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        return (hash1 >> 1) ^ hash2;
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
