#pragma once
#include "fst.hpp"
#include "beam_search.hpp"
#include "lattice.hpp"

class Decoder {
   public:
    using Path = vector<vector<string>>;
    struct SpecialSymbols {
        string startSymbol, endSymbol, epsSymbol;
    };


    Decoder(string graphFolder, string inputLabelsFile, SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
    Path decode(vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw);
    // vector<Path> getBestNPath(uint n);
    // void getLatticeWordSeqs(const Lattice& lattice, vector<vector<string>>& wordSeqs,const LatticeNode * root = NULL, vector<string>&& wordSeq = vector<string>());
    ~Decoder(){};

   private:
    Fst fst;
    BeamSearch beamSearch;
    unordered_map<string, uint> inpLabelToIndx;
    unordered_map<uint, uint> inpIdToActivationsIndx;
    SpecialSymbols espSyms;

    Path getBestPath();
    void parseInputLabels(const string& filename);
    void mapInpIdToActivationsIndx();
    void preprocessActivations(vector<vector<double>>& activations, double weight);
    void expandEpsStates();
    void applyFinalState();
    
};