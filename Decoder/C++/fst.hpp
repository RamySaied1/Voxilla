#pragma once
#include "helpers.hpp"

struct Arc {
    uint srcState, dstState, inpId, outId;
    double lmCost;
};

class Fst {
   public:
    Fst(string fstFile, string inpSymsTableFile,string outSymsTableFile,string epsSymbol="<eps>");
    ~Fst();

    bool hasEpsArc(uint state) { return graph[state].size() > 0 && inpSymsTable[graph[state].front()->inpId] == epsSymbol; }
    const vector<vector<const Arc*>>& getGraph() const { return graph; }
    const unordered_map<uint, double>& getFinalStates() const { return finalStates; }
    const unordered_map<uint, string> getInpSymsTable() const { return inpSymsTable; }
    const unordered_map<uint, string> getOutSymsTable() const { return outSymsTable; }

   private:
    vector<vector<const Arc*>> graph;
    vector<char*> inpLabels, outLabels;
    unordered_map<uint, double> finalStates;
    unordered_map<uint, string> inpSymsTable, outSymsTable;
    string epsSymbol;

    void parseFst(const string& fstFile);
    void parseSymsTable(const string& symsFile,unordered_map<uint, string>& symsTable);

    void processArc(const vector<string>& fields);
    void processFinalState(const vector<string>& fields);
    void preprocessFst();
};
