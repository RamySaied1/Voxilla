#include "decoder.hpp"

vector<vector<string>> Decoder::decode(vector<vector<double>>& activations, double amw) {
    preprocessActivations(activations, amw);  // normalize activations and apply lm relative weight
    unique_ptr<Arc> intialArc(new Arc{0, 0, fst.getSpecialSyms().epsSymbol, fst.getSpecialSyms().epsSymbol, 0.});
    beamSearch.setRootToken(intialArc.get(), 0., 0.);

    for (size_t i = 0; i < activations.size(); i++) {
        beamSearch.doForward(fst.getGraph(), fst.getInpLabelToIndx(), activations[i], true);
        beamSearch.beamPrune();
        expandEpsStates();

        beamSearch.moveExpandedToActive();
    }

    applyFinalState();
    return move(getBestPath());
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
        beamSearch.doForward(fst.getGraph(), {{fst.getSpecialSyms().epsSymbol, 0}}, vector<double>(1, 0.), false);
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

    // activeTokens.erase(iend, end(activeTokens));  // remove non final states
}

vector<vector<string>> Decoder::getBestPath() {
    Token finalToken = Token();
    auto path = beamSearch.getBestPath(fst.getGraph(), finalToken);
    // finalToken.print(cout);
    // cout << endl;
    // cout << "Combined Cost: " << (finalToken.amCost, finalToken.lmCost, finalToken.hmmCost);
    // cout << "\n---------------------------------\n";
    vector<vector<string>> inOutID(path.size(), vector<string>(3, ""));
    for (uint i = 0; i < path.size(); ++i) {
        const auto& arc = path[i];
        inOutID[i][0] = arc->inpLabel;
        inOutID[i][1] = arc->outLabel;
        inOutID[i][2] = to_string(arc->dstState);
    }
    return move(inOutID);
}