#include "decoder.hpp"

Decoder::Decoder(string graphFolder, string inputLabelsFile, SpecialSymbols espSyms) : beamSearch(), fst(graphFolder + "HCLG.txt", graphFolder + "input.syms", graphFolder + "output.syms", espSyms.epsSymbol), espSyms(espSyms) {
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
        inpIdToActivationsIndx[elem.first] = inpLabelToIndx[elem.second];
    }
}

Decoder::Path Decoder::decode(vector<vector<double>>& activations, uint maxActiveTokens, double beamWidth, double amw) {
    preprocessActivations(activations, amw);
    static unique_ptr<Arc> intialArc(new Arc{0, 0, 0, 0, 0.});  // dummy arc connected to intial state 0
    beamSearch.intiate(intialArc.get(), 0., 0., maxActiveTokens, beamWidth);

    for (size_t i = 0; i < activations.size(); i++) {
        beamSearch.doForward(fst.getGraph(), inpIdToActivationsIndx, activations[i], true);
        beamSearch.beamPrune();
        expandEpsStates();
        beamSearch.moveExpandedToActive();
    }

    applyFinalState();
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

void Decoder::expandEpsStates() {
    const vector<shared_ptr<Token>>& expandedTokens = beamSearch.getExpandedTokens();
    uint i = 0;
    while (i < expandedTokens.size()) {
        vector<shared_ptr<Token>> epsTokens;
        for (; i < expandedTokens.size(); ++i) {
            const auto& token = expandedTokens[i];
            if (fst.hasEpsArc(token->arc->dstState)) {
                epsTokens.push_back(token);
            }
        }
        beamSearch.setActiveTokens(epsTokens);
        beamSearch.doForward(fst.getGraph(), {{0, 0}}, vector<double>(1, 0.), false);  // eps symbol is assumed to have id 0
    }
    beamSearch.keepOnlyBestExpandedTokens();
}

void Decoder::applyFinalState() {
    auto activeTokens = beamSearch.getActiveTokens();
    auto iend = remove_if(begin(activeTokens), end(activeTokens), [&](const auto& t) {
        return fst.getFinalStates().find(t->arc->dstState) == fst.getFinalStates().end();  // remove if it's not a final state
    });

    for (auto i = begin(activeTokens); i != iend; ++i) {
        (*i)->lmCost += fst.getFinalStates().find((*i)->arc->dstState)->second;  // add final state cost
    }

    activeTokens.erase(iend, end(activeTokens));  // remove non final states
}

Decoder::Path Decoder::getBestPath() {
    Token finalToken = Token();
    auto path = beamSearch.getBestPath(finalToken);
    // finalToken.print(cout);
    // cout << endl;
    // cout << "Combined Cost: " << (finalToken.amCost, finalToken.lmCost, finalToken.hmmCost);
    // cout << "\n---------------------------------\n";
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
