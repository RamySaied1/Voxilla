
#pragma once
#include "beam_search.hpp"
#include "helpers.hpp"
struct Arc {
    int dstState;
    string inpLabel;
    string outLabel;
    double cost;
};

class Fst {
   public:
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };

    Fst(BeamSearch decoder, string fstFileName, string labelsFileName, SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
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

    void parseFst(const string& filename);
    void parseInputLabels(const string& filename);
    void processArc(const vector<string>& fields);
    void processFinalState(const vector<string>& fields);
    void preprocessActivations(vector<vector<double>>& activations, double relativeWeight);
    void preprocessFst();
    pair<int, double> skipStartNodes();
    void expandEpsStates();
};
