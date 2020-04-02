#include "decoder.hpp"

vector<vector<string>> Decoder::decode(vector<vector<double>>& activations, double amw, double lmw, double hmmw) {
    // assert(useLmWeightFrameFreq <= activations.size());
    // preprocessActivations(activations, amw);  // normalize activations and apply lm relative weight
    unique_ptr<Arc> intialArc(new Arc{0, 0, fst.getSpecialSyms().epsSymbol, fst.getSpecialSyms().epsSymbol, 0.});
    beamSearch.setRootToken(intialArc.get(), 0., 0., 0.);

    function<double(double, double, double)> combineAmLmHmmCosts = [&](double amCost, double lmCost, double hmmCost) { return amw*amCost + lmCost * lmw + hmmCost * hmmw; };
    for (size_t i = 0; i < activations.size(); i++) {
        // if (!i && i % useLmWeightFrameFreq == 0) {
        //     combineAmLmHmmCosts = [&](double amCost, double lmCost, double hmmCost) { return amCost; };
        // } else {
        //     combineAmLmHmmCosts = [&](double amCost, double lmCost, double hmmCost) { return amCost + lmw * lmCost; };
        // }
        beamSearch.doForward(fst.getGraph(), fst.getInpLabelToIndx(), activations[i], combineAmLmHmmCosts, true);
        beamSearch.beamPrune(combineAmLmHmmCosts);
        expandEpsStates(combineAmLmHmmCosts);

        beamSearch.moveExpandedToActive();
    }

    applyFinalState();
    return move(getBestPath(combineAmLmHmmCosts));
}

void Decoder::preprocessActivations(vector<vector<double>>& activations, double weight) {
    for (auto& exLogProbas : activations) {
        for (auto& elem : exLogProbas) {
            elem = (elem)*weight;
        }
    }
}

void Decoder::expandEpsStates(function<double(double, double, double)> combineAmLmHmmCosts) {
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
        beamSearch.doForward(fst.getGraph(), {{fst.getSpecialSyms().epsSymbol, 0}}, vector<double>(1, 0.), combineAmLmHmmCosts, false);
    }
    beamSearch.keepOnlyBestExpandedTokens(combineAmLmHmmCosts);
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

vector<vector<string>> Decoder::getBestPath(function<double(double, double, double)> combineAmLmHmmCosts) {
    Token finalToken = Token();
    auto path = beamSearch.getBestPath(fst.getGraph(), combineAmLmHmmCosts, finalToken);
    // finalToken.print(cout);
    // cout << endl;
    // cout << "Combined Cost: " << combineAmLmHmmCosts(finalToken.amCost, finalToken.lmCost, finalToken.hmmCost);
    // cout << "\n---------------------------------\n";
    vector<vector<string>> inOutID(path.size(), vector<string>(3, ""));
    for (uint i = 0; i < path.size(); ++i) {
        const auto& arc = path[i];
        inOutID[i][0] = arc->inpLabel;
        inOutID[i][1] = arc->outLabel;
        inOutID[i][2] = to_string(arc->srcState) + "->" + to_string(arc->dstState);
    }
    return move(inOutID);
}