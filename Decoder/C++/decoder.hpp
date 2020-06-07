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

    Decoder(string graphFolder, string inputLabelsFile, string grammerFileName = "G.fst", string fstFileName = "HCLG.txt", SpecialSymbols espSyms = {"<s>", "</s>", "<eps>"});
    Path decode(const vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam = 1);
    bool isSpecialSym(string sym) { return sym == espSyms.epsSymbol || sym == espSyms.startSymbol || sym == espSyms.endSymbol; }
    ~Decoder(){};

   private:
    unique_ptr<Fst> fst;
    BeamSearch beamSearch;
    unordered_map<string, uint> inpLabelToIndx;
    unordered_map<uint, uint> inpIdToActivationsIndx;
    SpecialSymbols espSyms;
    string graphFolder, inputLabelsFile;
    unique_ptr<fst::StdVectorFst> grammerFst;

    void parseInputLabels(const string& filename);
    void mapInpIdToActivationsIndx();
    void preprocessActivations(vector<vector<double>>& activations, double weight);
    void search(const vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam, const Fst* fst);
    void applyFinalState();
    void writeLatticeAsFst(string filename);
    Path getBestPath(const Fst* fst);
};