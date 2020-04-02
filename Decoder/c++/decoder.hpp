#include "beam_search.hpp"
#include "fst.hpp"

class Decoder {
   public:
    Decoder(string wfstFile, string labelsFile, int beamWidth, double pathAcceptingThreshold = 0) : beamSearch(beamWidth, pathAcceptingThreshold), fst(wfstFile, labelsFile) {}
    vector<vector<string>> decode(vector<vector<double>>& activations, double amw, double lmw, double hmmw );
    ~Decoder(){};

   private:
    Fst fst;
    BeamSearch beamSearch;
    void preprocessActivations(vector<vector<double>>& activations,double weight);
    void expandEpsStates(function<double(double, double,double)> combineAmLmHmmCosts);
    void applyFinalState();
    vector<vector<string>> getBestPath(function<double(double, double,double)> combineAmLmHmmCosts);
};