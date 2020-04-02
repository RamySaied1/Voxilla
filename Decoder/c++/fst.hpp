
#pragma once
#include "beam_search.hpp"
#include "helpers.hpp"
struct Arc {
    uint srcState, dstState;
    string inpLabel;
    string outLabel;
    double lmCost, transCost;
};

class Fst {
   public:
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };

    Fst(string fstFileName, string labelsFileName, SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    ~Fst();

    bool hasEpsArc(uint state) { return graph[state].size() > 0 && graph[state].front()->inpLabel == espSyms.epsSymbol; }
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
    const SpecialSymbols& getSpecialSyms() const {return espSyms;}
    const vector<vector<const Arc*>>& getGraph() const {return graph;}
    const unordered_map<string, uint>& getInpLabelToIndx() const {return inpLabelToIndx;}
    const unordered_map<uint, double>& getFinalStates() const {return finalStates;}

   private:
    struct TransitionInfo {
        string inpLabel;
        double transProba;
    };

    vector<vector<const Arc*>> graph;
    unordered_map<string, uint> inpLabelToIndx;
    unordered_map<uint, TransitionInfo> transIdToTransitionInfo;
    unordered_map<uint, double> finalStates;
    SpecialSymbols espSyms;
    pair<double,double> initialMinMaxArcCost,newMinMaxArcCost;

    void parseFst(const string& filename);
    void parseInputLabels(const string& filename);
    void parseTransitionsInfo(const string& filename);
    void processArc(const vector<string>& fields);
    void processFinalState(const vector<string>& fields);
    void preprocessFst();
};
