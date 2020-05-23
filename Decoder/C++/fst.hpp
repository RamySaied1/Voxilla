#pragma once
#include "helpers.hpp"
struct Arc {
    uint srcState, dstState, inpId, outId;
    double lmCost;
};

struct ArcComparator {
    bool operator()(const Arc& a1, const Arc& a2) const {
        if (a1.srcState != a2.srcState) return a1.srcState < a2.srcState;
        if (a1.dstState != a2.dstState) return a1.dstState < a2.dstState;
        if (a1.inpId != a2.inpId) return a1.inpId < a2.inpId;
        if (a1.outId != a2.outId) return a1.outId < a2.outId;
        // if (abs(a1.lmCost - a2.lmCost) > 1e-4) return a1.lmCost < a2.lmCost; // this is disabled on purpose
    }
};

class Fst {
   public:
    Fst(string fstFile, string inpSymsTableFile, string outSymsTableFile, string epsSymbol = "<eps>");
    ~Fst();

    bool hasEpsArc(uint state) const { return graph[state].size() > 0 && inpSymsTable.find(graph[state].front()->inpId)->second == epsSymbol; }
    const vector<vector<const Arc*>>& getGraph() const { return graph; }
    const unordered_map<uint, double>& getFinalStates() const { return finalStates; }
    const unordered_map<uint, string> getInpSymsTable() const { return inpSymsTable; }
    const unordered_map<uint, string> getOutSymsTable() const { return outSymsTable; }
    string getEpsSymbol() const { return epsSymbol; }

   private:
    vector<vector<const Arc*>> graph;
    vector<char*> inpLabels, outLabels;
    unordered_map<uint, double> finalStates;
    unordered_map<uint, string> inpSymsTable, outSymsTable;
    string epsSymbol;

    void parseFst(const string& fstFile);
    void parseSymsTable(const string& symsFile, unordered_map<uint, string>& symsTable);

    void processArc(const vector<string>& fields);
    void processFinalState(const vector<string>& fields);
    void preprocessFst();
};
