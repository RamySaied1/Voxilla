#include "fst.hpp"
#include "beam_search.hpp"

class Decoder {
   private:
    Fst fst;

   public:
    Decoder(string wfstFile, string labelsFile, int beamWidth, double pathAcceptingThreshold = 0) : fst(BeamSearch(beamWidth, pathAcceptingThreshold), wfstFile, labelsFile) {
    }

    vector<vector<string>> decode(vector<vector<double>>& activations, double lmWeigh) {
        vector<const Arc*> path = fst.decode(activations, 30.);
        vector<vector<string>> inOut(path.size(), vector<string>(2, ""));
        for (uint i = 0; i < path.size(); ++i) {
            const auto& arc = path[i];
            inOut[i][0] = arc->inpLabel;
            inOut[i][1] = arc->outLabel;
        }
        return move(inOut);
    }

    ~Decoder(){};
};