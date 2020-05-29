#include "decoder.hpp"

Decoder::Decoder(string graphFolder, string inputLabelsFile, string fstFileName, SpecialSymbols espSyms) : beamSearch(), fst(graphFolder + fstFileName, graphFolder + "input.syms", graphFolder + "output.syms", espSyms.epsSymbol), espSyms(espSyms) {
    inpIdToActivationsIndx = unordered_map<uint, uint>();
    parseInputLabels(inputLabelsFile);
    mapInpIdToActivationsIndx();
}

void Decoder::parseInputLabels(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);
    if (!in.is_open()) {
        throw Exception("Can't open input labels file: " + filename);
    }
    string line;
    uint i = 0;
    while (in >> line) {
        strtolower(line);
        if (isSpecialSym(line)) {
            throw Exception("special symbol exists in labels file which is not expected");
        } else {
            inpLabelToIndx[line] = i++;
        }
    }
}

void Decoder::mapInpIdToActivationsIndx() {
    for (const auto& elem : fst.getInpSymsTable()) {
        if (elem.second != fst.getEpsSymbol()) {
            inpIdToActivationsIndx[elem.first] = inpLabelToIndx[elem.second];
        }
    }
}

Decoder::Path Decoder::decode(vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam) {
    preprocessActivations(activations, amw);
    static unique_ptr<Arc> intialArc(new Arc{0, 0, 0, 0, 0.});  // dummy arc connected to intial state 0
    beamSearch.intiate(intialArc.get(), 0., 0., maxActiveTokens, beamWidth, latticeBeam, &fst);

    for (size_t i = 0; i < activations.size(); i++) {
        beamSearch.startNewExpantions();                                     // start new time step
        beamSearch.doForward(inpIdToActivationsIndx, activations[i], true);  // expand the frontier
        beamSearch.createExpandedTokens();                                   // create the expanded tokens from frontier
        beamSearch.beamPrune();                                              // prune and only keep bet tokens
        beamSearch.expandEpsStates();                                        // expand epslions
        beamSearch.moveExpandedToActive();                                   // move expaned tokens to actve to be ready for new timestep
        beamSearch.finishExpantions();                                       // end time step
    }

    beamSearch.finalize();

    static uint uniqueNum = 0;
    uniqueNum++;
    string filename = "lat" + to_string(uniqueNum) + ".txt";
    string folder = "./C++/Lattices/";
    const unordered_map<uint, double>& finalStates = fst.getFinalStates();
    beamSearch.getLattice()->writeAsFst(folder + filename, beamSearch.getActiveTokens(), finalStates);

    return getBestPath();
}

void Decoder::preprocessActivations(vector<vector<double>>& activations, double weight) {
    for (auto& exLogProbas : activations) {
        double maxVal = *max_element(begin(exLogProbas), end(exLogProbas));
        for (auto& elem : exLogProbas) {
            elem = (elem - maxVal) * weight;
        }
    }
}

Decoder::Path Decoder::getBestPath() {
    double pathCost;
    auto path = beamSearch.getBestPath(pathCost);
    cout<<"Total Path Cost: "<<pathCost<<endl;
    auto inpSymsTable = fst.getInpSymsTable();
    auto outSymsTable = fst.getOutSymsTable();
    Path inOutID(path.size(), vector<string>(3, ""));
    for (uint i = 0; i < path.size(); ++i) {
        const auto& arc = path[i];
        inOutID[i][0] = inpSymsTable.find(arc->inpId)->second;
        inOutID[i][1] = outSymsTable.find(arc->outId)->second;
        inOutID[i][2] = to_string(arc->dstState);
    }
    return inOutID;
}
