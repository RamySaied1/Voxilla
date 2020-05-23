#pragma once
#include "beam_search.hpp"
#include "fst.hpp"
#include "lattice.hpp"

class Decoder {
   public:
    using Path = vector<vector<string>>;
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };

    Decoder(string graphFolder, string inputLabelsFile, string fstFileName = "HCLG.txt", SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    Path decode(vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam);
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
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
    void applyFinalState();
    Path getBestPath();
};