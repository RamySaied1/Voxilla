#pragma once
#include "fst.hpp"

class Decoder {
   public:
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };
    using Path = vector<vector<string>>; 

    Decoder(string graphFolder, string inputLabelsFile, SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
    vector<vector<string>> decode(vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw);
    vector<Path> getBestNPath(uint i);
    ~Decoder(){};

   private:
    Fst fst;
    BeamSearch beamSearch;
    unordered_map<string, uint> inpLabelToIndx;
    unordered_map<uint, uint> inpIdToActivationsIndx;
    SpecialSymbols espSyms;

    void parseInputLabels(const string& filename);
    void mapInpIdToActivationsIndx();
    void preprocessActivations(vector<vector<double>>& activations, double weight);
    void expandEpsStates();
    void applyFinalState();
    Path getBestPath();
};