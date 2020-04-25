#include "fst.hpp"

#include <exception>

Fst::Fst(string fstFile, string inpSymsTableFile, string outSymsTableFile,string epsSymbol): epsSymbol(epsSymbol) {
    graph = vector<vector<const Arc*>>();
    finalStates = unordered_map<uint, double>();
    parseFst(fstFile);
    parseSymsTable(inpSymsTableFile,inpSymsTable);
    parseSymsTable(outSymsTableFile,outSymsTable);
    preprocessFst();
}

void Fst::parseFst(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);
    if (!in.is_open()) {
        throw Exception("Can't open Fst file: " + filename);
    }
    try {
        string line;
        while (getline(in, line)) {
            vector<string> fields;
            split(line, fields);

            if (fields.size() <= 2) {
                processFinalState(fields);
            } else {
                processArc(fields);
            }
        }
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        graph.clear();
    }
    in.close();
}

void Fst::parseSymsTable(const string& symsFile, unordered_map<uint, string>& symsTable) {
    ifstream in;
    in.open(symsFile, ifstream::in);
    if (!in.is_open()) {
        throw Exception("Can't open Syms file: " + symsFile);
    }

    string label, id;
    while (in >> label) {
        tolower(label);
        in >> id;
        symsTable[stoi(id)] = label;
    }
}

void Fst::processFinalState(const vector<string>& fields) {
    uint finalState = (uint)stoi(fields[0]);
    double cost = (fields.size() == 2) ? -stod(fields.back()) : 0.;
    finalStates[finalState] = cost;
    if (finalState == graph.size()) {
        graph.push_back(vector<const Arc*>());
    }
}

void Fst::processArc(const vector<string>& fields) {
    uint srcState = (uint)stoi(fields[0]);
    double lmCost = (fields.size() == 5) ? -stod(fields.back()) : 0.;
    const Arc* arc = new Arc {srcState, (uint)stoi(fields[1]), (uint)stoi(fields[2]), (uint)stoi(fields[3]), lmCost};

    if (srcState < graph.size()) {
        graph[srcState].push_back(arc);
    } else {
        graph.push_back(vector<const Arc*>({arc}));
    }
}

void Fst::preprocessFst() {
    graph.reserve(graph.size());
    for (auto& Arcs : graph) {
        moveRelevantFisrt<const Arc*>(Arcs, [&](auto arc) {  // put speical symbols at first
            return arc->inpId == 0;
        });
        Arcs.reserve(Arcs.size());  // reclaim useless space
    }
}

Fst::~Fst() {
    for (auto& arcs : graph) {
        for (auto& arc : arcs) {
            delete arc;
        }
    }
}