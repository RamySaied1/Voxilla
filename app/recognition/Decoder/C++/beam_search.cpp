#include "beam_search.hpp"

BeamSearch::BeamSearch(uint maxActiveTokens, double beamWidth) : maxActiveTokens(maxActiveTokens), beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()) {
    assert(maxActiveTokens != 0);
}

void BeamSearch::intiate(const Arc* arc, double lmCost, double amCost, uint maxActiveTokens, double beamWidth, uint latticeBeam, const Fst* fst) {
    this->maxActiveTokens = maxActiveTokens;
    this->beamWidth = beamWidth;
    this->fst = fst;

    activeTokens.clear();
    expandedTokens = {shared_ptr<Token>(new Token{0, arc, 0., 0., 0.})};  // set with intial token
    lattice = Lattice(latticeBeam, expandedTokens.back());

    lattice.startNewExpantions();
    expandEpsStates();
    lattice.finishExpantions(beamWidth);
    moveExpandedToActive();
}

void BeamSearch::doForward(const unordered_map<uint, uint>& inpIdsToIndx, const vector<double>& activations, bool useSelfLoops) {
    const vector<vector<const Arc*>>& graph = fst->getGraph();
    if (useSelfLoops) {
        for (uint i = 0; i < activeTokens.size(); ++i) {
            const auto& token = activeTokens[i];
            auto iter = inpIdsToIndx.find(token->arc->inpId);
            if (iter == inpIdsToIndx.end()) continue;
            double amCost = activations[iter->second];
            lattice.expand(token, token->arc, 0., amCost);
        }
    }

    for (uint i = 0; i < activeTokens.size(); ++i) {
        const auto& token = activeTokens[i];
        for (const Arc* arc : graph[token->arc->dstState]) {
            if (token->arc->dstState == arc->dstState) {
                throw Exception("SELF LOOP DETECTED");  // graph should has no self loops
            }
            auto iter = inpIdsToIndx.find(arc->inpId);
            if (iter == inpIdsToIndx.end()) continue;
            double lmCost = arc->lmCost;
            double amCost = activations[iter->second];

            //Expand the frontier
            lattice.expand(token, arc, lmCost, amCost);
        }
    }
}

void BeamSearch::expandEpsStates() {
    uint i = 0;
    while (i < expandedTokens.size()) {
        vector<shared_ptr<Token>> epsTokens;
        for (; i < expandedTokens.size(); ++i) {
            const auto& token = expandedTokens[i];
            if (fst->hasEpsArc(token->arc->dstState)) {
                epsTokens.push_back(token);
            }
        }
        setActiveTokens(epsTokens);
        doForward({{0, 0}}, vector<double>(1, 0.), false);  // eps symbol is assumed to have id 0
        createExpandedTokens();
    }
    keepOnlyBestExpandedTokens();
}

void BeamSearch::keepOnlyBestExpandedTokens() {
    const unordered_map<const Arc*, shared_ptr<Token>>& arcToBestToken = lattice.getArcToToken();
    expandedTokens.erase(remove_if(begin(expandedTokens), end(expandedTokens), [&](const auto& token) { return arcToBestToken.find(token->arc) != arcToBestToken.end() && arcToBestToken.find(token->arc)->second != token; }), end(expandedTokens));
}

void BeamSearch::setActiveTokens(const vector<shared_ptr<Token>>& tokens) {
    activeTokens = tokens;
}

const vector<shared_ptr<Token>>& BeamSearch::getExpandedTokens() const {
    return expandedTokens;
}

const vector<shared_ptr<Token>>& BeamSearch::getActiveTokens() const {
    return activeTokens;
}

void BeamSearch::moveExpandedToActive() {
    activeTokens = move(expandedTokens);
}

void BeamSearch::beamPrune() {
    if (expandedTokens.size() > maxActiveTokens) {
        sort(begin(expandedTokens), end(expandedTokens), [&](const shared_ptr<Token>& a, const shared_ptr<Token>& b) {
            return a->amCost + a->lmCost > b->amCost + b->lmCost;
        });
        // remove pruned tokens form lattice to reclaim memory
        for (uint i = maxActiveTokens; i < expandedTokens.size(); ++i) {
            lattice.removeToken(expandedTokens[i]);
        }
        expandedTokens.resize(maxActiveTokens);
    }
}

vector<const Arc*> BeamSearch::getBestPath(double& pathCost) {
    if (activeTokens.size() <= 0) return vector<const Arc*>();
    const unordered_map<uint, double>& finalStates = fst->getFinalStates();
    auto bestToken = *max_element(begin(activeTokens), end(activeTokens), [&](const auto& t1, const auto& t2) {
        return t1->amCost + t1->lmCost + finalStates.find(t1->arc->dstState)->second <
               t2->amCost + t2->lmCost + finalStates.find(t2->arc->dstState)->second;
    });
    pathCost = bestToken->amCost + bestToken->lmCost + finalStates.find(bestToken->arc->dstState)->second;
    return lattice.getBestPath(bestToken);
}

void BeamSearch::finalize() {
    const unordered_map<uint, double>& finalStates = fst->getFinalStates();
    auto iend = remove_if(begin(activeTokens), end(activeTokens), [&](const auto& t) {
        return finalStates.find(t->arc->dstState) == finalStates.end();  // remove if it's not a final state
    });

    uint newSize = iend - begin(activeTokens);
    for (uint i = newSize; i < expandedTokens.size(); ++i) {
        lattice.removeToken(activeTokens[i]);
    }
    activeTokens.resize(newSize);  // remove non final states
}

BeamSearch::~BeamSearch() {}
