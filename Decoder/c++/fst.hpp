
#pragma once
#include "beam_search.hpp"
#include "helpers.hpp"
struct Arc {
    uint srcState, dstState;
    string inpLabel;
    string outLabel;
    double lmCost;
};

class Fst {
   public:
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };

    Fst(BeamSearch decoder, string fstFileName, string labelsFileName,string minmaxArcCostFileName,pair<double,double> newMinMaxArcCost={-3.,0.}, SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    ~Fst();

    bool hasEpsArc(uint state) { return graph[state].size() > 0 && graph[state].front()->inpLabel == espSyms.epsSymbol; }
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
    vector<const Arc*> decode(vector<vector<double>>& activations, double lmWeight);

   private:
    vector<vector<const Arc*>> graph;
    unordered_map<string, uint> inpLabelToIndx;
    unordered_map<uint, double> finalStates;
    SpecialSymbols espSyms;
    BeamSearch decoder;
    pair<double,double> initialMinMaxArcCost,newMinMaxArcCost;

    void parseFst(const string& fstFilename,const string& minmaxFilename);
    void parseInputLabels(const string& filename);
    void processArc(const vector<string>& fields);
    void processFinalState(const vector<string>& fields);
    void preprocessActivations(vector<vector<double>>& activations, double relativeWeight);
    void preprocessFst();
    void expandEpsStates();
    void applyFinalState();
};
