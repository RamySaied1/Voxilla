#include "decoder.hpp"

Decoder::Decoder(string graphFolder, string inputLabelsFile, string grammerFileName, string fstFileName, SpecialSymbols espSyms) : beamSearch(), fst(new Fst(graphFolder + fstFileName, graphFolder + "input.syms", graphFolder + "output.syms", espSyms.epsSymbol)), graphFolder(graphFolder), inputLabelsFile(inputLabelsFile), espSyms(espSyms), grammerFst(nullptr) {
    inpIdToActivationsIndx = unordered_map<uint, uint>();
    parseInputLabels(inputLabelsFile);
    mapInpIdToActivationsIndx();
    grammerFst = grammerFileName != "" ? unique_ptr<fst::StdVectorFst>(fst::StdVectorFst::Read(graphFolder + grammerFileName)) : unique_ptr<fst::StdVectorFst>(nullptr);
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
    for (const auto& elem : fst->getInpSymsTable()) {
        if (elem.second != fst->getEpsSymbol()) {
            inpIdToActivationsIndx[elem.first] = inpLabelToIndx[elem.second];
        }
    }
}

Decoder::Path Decoder::decode(const vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam) {
    vector<vector<double>> normalizedActivations = activations;
    preprocessActivations(normalizedActivations, amw);

    search(normalizedActivations, maxActiveTokens, beamWidth, amw, latticeBeam, fst.get());

    if (latticeBeam > 1 && grammerFst.get() != nullptr) {
        writeLatticeAsFst(graphFolder + "lat.txt");
        unique_ptr<Fst> newFst(new Fst(graphFolder + "lat.txt", graphFolder + "input.syms", graphFolder + "output.syms", espSyms.epsSymbol));
        search(normalizedActivations, 1000, 25, amw, 1, newFst.get());
        return getBestPath(newFst.get());
    }

    return getBestPath(fst.get());
}

void Decoder::preprocessActivations(vector<vector<double>>& activations, double weight) {
    for (auto& exLogProbas : activations) {
        double maxVal = *max_element(begin(exLogProbas), end(exLogProbas));
        for (auto& elem : exLogProbas) {
            elem = (elem - maxVal) * weight;
        }
    }
}

void Decoder::search(const vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw, uint latticeBeam, const Fst* fst) {
    static unique_ptr<Arc> intialArc(new Arc{0, 0, 0, 0, 0.});  // dummy arc connected to intial state 0
    beamSearch.intiate(intialArc.get(), 0., 0., maxActiveTokens, beamWidth, latticeBeam, fst);

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
}

void Decoder::writeLatticeAsFst(string filename) {
    unique_ptr<fst::StdVectorFst> latFst(new fst::StdVectorFst());
    beamSearch.getLattice()->latticToFst(beamSearch.getActiveTokens(), fst->getFinalStates(), latFst.get());
    fst::ArcMap(latFst.get(), fst::RmWeightMapper<fst::StdArc>());
    fst::ArcSort(latFst.get(), fst::StdOLabelCompare());
    fst::StdVectorFst result;
    fst::Compose(*latFst, *grammerFst, &result);
    writeFst(result, filename);
}

Decoder::Path Decoder::getBestPath(const Fst* fst) {
    double pathCost;
    auto path = beamSearch.getBestPath(pathCost);
    cout << "Total Path Cost: " << pathCost << endl;
    auto inpSymsTable = fst->getInpSymsTable();
    auto outSymsTable = fst->getOutSymsTable();
    Path inOutID(path.size(), vector<string>(3, ""));
    for (uint i = 0; i < path.size(); ++i) {
        const auto& arc = path[i];
        inOutID[i][0] = inpSymsTable.find(arc->inpId)->second;
        inOutID[i][1] = outSymsTable.find(arc->outId)->second;
        inOutID[i][2] = to_string(arc->dstState);
    }
    return inOutID;
}
