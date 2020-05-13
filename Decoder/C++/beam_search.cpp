#include "beam_search.hpp"

BeamSearch::BeamSearch(uint maxActiveTokens, double beamWidth) : maxActiveTokens(maxActiveTokens), beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()) {
    assert(maxActiveTokens != 0);
}

void BeamSearch::intiate(const Arc* arc, double lmCost, double amCost, uint maxActiveTokens, double beamWidth) {
    this->maxActiveTokens = maxActiveTokens;
    this->beamWidth = beamWidth;

    expandedTokens.clear();
    activeTokens.clear();
    lattice = Lattice();

    lattice.startNewExpantions();
    lattice.expand(nullptr, arc, lmCost, amCost);
    lattice.finishExpantions(expandedTokens);
    moveExpandedToActive();
}

void BeamSearch::keepOnlyBestExpandedTokens() {
    unordered_map<const Arc*, shared_ptr<Token>> arcToBestToken;
    for (const auto& token : expandedTokens) {
        double tokenCost = token->amCost + token->lmCost;
        auto iter = arcToBestToken.find(token->arc);
        if (iter == arcToBestToken.end()) {
            arcToBestToken[token->arc] = token;
        } else {
            if (tokenCost > iter->second->amCost + iter->second->lmCost) {
                iter->second = token;
            }
        }
    }

    expandedTokens.erase(remove_if(begin(expandedTokens), end(expandedTokens), [&](const auto& t) { return arcToBestToken[t->arc] != t; }),
                         end(expandedTokens));
}

void BeamSearch::doForward(const vector<vector<const Arc*>>& graph, const unordered_map<uint, uint>& inpIdsToIndx, const vector<double>& activations, bool useSelfLoops) {
    lattice.startNewExpantions();  // start new expantions stage

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
                continue;  // graph has no self loops by defualt but this is sainty check
            }
            auto iter = inpIdsToIndx.find(arc->inpId);
            if (iter == inpIdsToIndx.end()) continue;
            double lmCost = arc->lmCost;
            double amCost = activations[iter->second];

            //Expand the frontier
            lattice.expand(token, arc, lmCost, amCost);
        }
    }

    // get new created tokens
    lattice.finishExpantions(expandedTokens);
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
    if (expandedTokens.size() > maxActiveTokens / 2) {
        sort(begin(expandedTokens), end(expandedTokens), [&](const shared_ptr<Token>& a, const shared_ptr<Token>& b) {
            return a->amCost + a->lmCost > b->amCost + b->lmCost;
        });

        double thresold = expandedTokens.front()->amCost + expandedTokens.front()->lmCost - beamWidth;
        auto iter = lower_bound(rbegin(expandedTokens), rend(expandedTokens), thresold, [&](const shared_ptr<Token>& t, double val) {
            return t->amCost + t->lmCost < val;
        });
        // cout<<"Original Size: "<<expandedTokens.size()<<endl;
        uint newSize = rend(expandedTokens) - iter;
        newSize = min(newSize, maxActiveTokens);
        for (int i = newSize; i < expandedTokens.size(); ++i) {
            lattice.removeToken(expandedTokens[i]);
        }
        expandedTokens.resize(newSize);
        // cout<<"New Size due to beam: "<<expandedTokens.size()<<endl;
    }
}

vector<const Arc*> BeamSearch::getBestPath(Token& finalToken) {
    if (activeTokens.size() <= 0) return vector<const Arc*>();

    auto bestToken = *max_element(begin(activeTokens), end(activeTokens), [&](const auto& t1, const auto& t2) {
        return t1->amCost + t1->lmCost < t2->amCost + t2->lmCost;
    });
    finalToken = *(bestToken);

    vector<shared_ptr<Token>> tokensSeq = lattice.getBestPath(bestToken);

    vector<const Arc*> arcs = vector<const Arc*>(tokensSeq.size());
    transform(begin(tokensSeq), end(tokensSeq), begin(arcs), [](const auto& token) {
        return token->arc;
    });
    return arcs;
}

void BeamSearch::applyFinalState(const unordered_map<uint, double>& finalStates) {
    auto iend = remove_if(begin(activeTokens), end(activeTokens), [&](const auto& t) {
        return finalStates.find(t->arc->dstState) == fst.getFinalStates().end();  // remove if it's not a final state
    });

    for (auto i = begin(activeTokens); i != iend; ++i) {
        (*i)->lmCost += finalStates.find((*i)->arc->dstState)->second;  // add final state cost
    }

    uint newSize = iend - begin(activeTokens);
    for (int i = newSize; i < activeTokens.size(); ++i) {
        lattice.removeToken(activeTokens[i]);
    }
    activeTokens.resize(newSize); // remove non final states
}

// vector<double> BeamSearch::getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens) {
//     if (tokens.size() <= 0) return vector<double>();
//     const auto& bestToken = *max_element(begin(tokens), end(tokens), [&](const auto& t1, const auto& t2) {
//         return t1->amCost + t1->lmCost < t2->amCost + t2->lmCost;
//     });
//     double bestCost = bestToken->amCost + bestToken->lmCost;

//     vector<double> logProbas(tokens.size(), 0);
//     for (uint i = 0; i < tokens.size(); ++i) {
//         logProbas[i] = tokens[i]->amCost + tokens[i]->lmCost - bestCost;
//     }
//     return logProbas;
// }

// vector<vector<const Arc*>> BeamSearch::getBestNPath(uint n) {
//     n = min(n, (uint)activeTokens.size());
//     vector<vector<const Arc*>> pathes = vector<vector<const Arc*>>(n);
//     if (n <= 0) return pathes;

//     vector<shared_ptr<Token>> mockActiveTokens(begin(activeTokens), end(activeTokens));

//     // // remove dupliceates
//     // sort(begin(mockActiveTokens), end(mockActiveTokens), [&](const auto& t1, const auto& t2) {
//     //     return t1->arc < t2->arc;
//     // });
//     // mockActiveTokens.erase(unique(mockActiveTokens.begin(), mockActiveTokens.end(),[&](const shared_ptr<Token>& t1, const shared_ptr<Token>& t2) {
//     //     return t1->arc == t2->arc;
//     // }), mockActiveTokens.end());

//     // sort by cost
//     sort(begin(mockActiveTokens), end(mockActiveTokens), [&](const auto& t1, const auto& t2) {
//         return t1->amCost + t1->lmCost > t2->amCost + t2->lmCost;
//     });
//     // int cd = 0;
//     // for (int i = 1; i < mockActiveTokens.size(); ++i) {
//     //     if (mockActiveTokens[i]->amCost + mockActiveTokens[i]->lmCost == mockActiveTokens[i - 1]->amCost + mockActiveTokens[i - 1]->lmCost) {
//     //         ++cd;
//     //     }
//     // }

//     for (uint i = 0; i < n; ++i) {
//         shared_ptr<Token> currToken = mockActiveTokens[i];
//         while (currToken) {
//             pathes[i].push_back(currToken->arc);
//             currToken = predeccessor[currToken];
//         }
//         reverse(begin(pathes[i]), end(pathes[i]));
//     }
//     return pathes;
// }

BeamSearch::~BeamSearch() {}