#include "beam_search.hpp"

BeamSearch::BeamSearch(uint maxActiveTokens, double beamWidth) : maxActiveTokens(maxActiveTokens), beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()) {
    assert(maxActiveTokens != 0);
}

void BeamSearch::intiate(const Arc* arc, double lmCost, double amCost, uint maxActiveTokens, double beamWidth, uint latticeBeam, const Fst* fst) {
    this->maxActiveTokens = maxActiveTokens;
    this->beamWidth = beamWidth;
    this->fst = fst;

    expandedTokens.clear();
    activeTokens.clear();
    lattice = Lattice(latticeBeam);

    lattice.startNewExpantions();
    lattice.expand(shared_ptr<Token>(nullptr), arc, lmCost, amCost);
    lattice.createExpandedTokens(expandedTokens, beamWidth);
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

vector<const Arc*> BeamSearch::getBestPath(Token& finalToken) {
    if (activeTokens.size() <= 0) return vector<const Arc*>();

    auto bestToken = *max_element(begin(activeTokens), end(activeTokens), [&](const auto& t1, const auto& t2) {
        return t1->amCost + t1->lmCost < t2->amCost + t2->lmCost;
    });
    finalToken = *(bestToken);

    return lattice.getBestPath(bestToken);
}

void BeamSearch::applyFinalState(const unordered_map<uint, double>& finalStates) {
    auto iend = remove_if(begin(activeTokens), end(activeTokens), [&](const auto& t) {
        return finalStates.find(t->arc->dstState) == finalStates.end();  // remove if it's not a final state
    });

    for (auto i = begin(activeTokens); i != iend; ++i) {
        (*i)->lmCost += finalStates.find((*i)->arc->dstState)->second;  // add final state cost
    }

    uint newSize = iend - begin(activeTokens);
    for (uint i = newSize; i < expandedTokens.size(); ++i) {
        lattice.removeToken(activeTokens[i]);
    }
    activeTokens.resize(newSize);  // remove non final states
}

BeamSearch::~BeamSearch() {}