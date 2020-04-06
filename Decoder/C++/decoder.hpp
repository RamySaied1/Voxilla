#include "beam_search.hpp"
#include "fst.hpp"

class Decoder {
   public:
    Decoder(string wfstFile, string labelsFile,string hmmFile, uint maxActiveTokens, double beamWidth = 10.) : beamSearch(maxActiveTokens, beamWidth), fst(wfstFile, labelsFile,hmmFile) {}
    vector<vector<string>> decode(vector<vector<double>>& activations, double amw);
    ~Decoder(){};

   private:
    Fst fst;
    BeamSearch beamSearch;
    void preprocessActivations(vector<vector<double>>& activations,double weight);
    void expandEpsStates();
    void applyFinalState();
    vector<vector<string>> getBestPath();
};