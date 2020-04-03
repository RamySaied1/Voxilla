#include "fst.hpp"
#include <exception>

Fst::Fst(string fstFileName, string labelsFileName, SpecialSymbols espSyms) : espSyms(espSyms) {
    inpLabelToIndx = unordered_map<string, uint>();
    parseInputLabels(labelsFileName);
    parseTransitionsInfo("../hmm.txt");

    graph = vector<vector<const Arc*>>();
    finalStates = unordered_map<uint, double>();
    parseFst(fstFileName);
    preprocessFst();
}

void Fst::parseInputLabels(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);
    if (!in.is_open()) {
        throw Exception("Can't open input labels file: " + filename);
    }
    string line;
    uint i = 0;
    while (in >> line) {
        if (isSpecialSym(line)) {
            throw Exception("special symbol exists in labels file which is not expected");
        } else {
            inpLabelToIndx[line] = i++;
        }
    }
}

void Fst::parseTransitionsInfo(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);
    if (!in.is_open()) {
        throw Exception("Can't open TransitionsInfo file: " + filename);
    }
    string line, currPhone, hmmState, transitionSate;
    while (getline(in, line)) {
        vector<string> fields;
        split(line, fields);

        if (fields.size() == 11) {
            transitionSate = fields[1].substr(0, fields[1].size() - 1);
            currPhone = fields[4];
            hmmState = fields[7];
            assert(inpLabelToIndx.find(currPhone + "_s" + hmmState) != inpLabelToIndx.end());
        } else if (fields.size() > 5) {
            string nextState = currPhone + "_s";
            nextState += (fields.size() > 7) ? fields[6].substr(1, fields[6].size() - 1) : hmmState;
            uint transId = stoi(fields[2]);
            transIdToInpLabel[transId] = nextState;
            // cout << nextState << "\t" << transId << endl;
        } else {
            cout << fields.size() << endl;
            throw Exception("File format error -> wrong number of tokens");
        }
    }
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
    uint transId = stoi(fields[2]);
    string inpLabel = transIdToInpLabel[transId];
    double lmCost = (fields.size() == 5) ? -stod(fields.back()) : 0.;
    const Arc* arc = new Arc{srcState, (uint)stoi(fields[1]), inpLabel, fields[3], lmCost};

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
            return arc->inpLabel == espSyms.epsSymbol;
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