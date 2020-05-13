#include "beam_search.hpp"
#include "fst.hpp"

BeamSearch::BeamSearch(uint maxActiveTokens, double beamWidth) : maxActiveTokens(maxActiveTokens), beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
    assert(maxActiveTokens != 0);
}

void BeamSearch::intiate(const Arc* arc, double lmCost, double amCost, uint maxActiveTokens, double beamWidth) {
    this->maxActiveTokens = maxActiveTokens;
    this->beamWidth = beamWidth;

    expandedTokens.clear();
    activeTokens.clear();
    predeccessor.clear();
    expandNewToken(arc, lmCost, amCost);
    predeccessor[expandedTokens.front()] = shared_ptr<Token>(NULL);
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
    unordered_map<const Arc*, Expantion> expantions;  // map expanded node to parent node and expantion cost
    vector<double> logProbas = getNormalizeTokensLogProba(activeTokens);
    if (useSelfLoops) {
        for (uint i = 0; i < activeTokens.size(); ++i) {
            const auto& token = activeTokens[i];
            auto iter = inpIdsToIndx.find(token->arc->inpId);
            if (iter == inpIdsToIndx.end()) continue;
            // double lmCost = token->arc->lmCost;
            double amCost = activations[iter->second];
            // double specialCost = (token->arc->srcState == token->arc->dstState) ? log(2.) : 0;
            double expantionCost = logProbas[i];
            expantions[token->arc] = Expantion(token, 0., amCost, expantionCost);
        }
    }

    for (uint i = 0; i < activeTokens.size(); ++i) {
        const auto& token = activeTokens[i];
        for (const Arc* arc : graph[token->arc->dstState]) {
            auto iter = inpIdsToIndx.find(arc->inpId);
            if (iter == inpIdsToIndx.end()) continue;
            double lmCost = arc->lmCost;
            double amCost = activations[iter->second];

            //Expand the frontier and add predecessors
            double expantionCost = logProbas[i] + lmCost;
            if (exp(expantionCost) > 0.) {
                bool isNewArc = expantions.find(arc) == expantions.end();
                if (isNewArc) {
                    expantions[arc] = Expantion(token, lmCost, amCost, expantionCost);
                } else {
                    double oldCost = expantions[arc].expantionCost;
                    if (expantionCost > oldCost) {
                        expantions[arc] = Expantion(token, lmCost, amCost, expantionCost);
                    }
                }
            }
        }
    }

    // create new tokens
    createExpandedTokens(expantions);
}

void BeamSearch::createExpandedTokens(const unordered_map<const Arc*, Expantion>& expantions) {
    for (const auto& keyVal : expantions) {
        const auto& expantion = keyVal.second;
        const Arc* arc = keyVal.first;

        double amCost = expantion.parentToken->amCost + expantion.amCost;
        double lmCost = expantion.parentToken->lmCost + expantion.lmCost;

        expandNewToken(arc, lmCost, amCost);
        predeccessor[expandedTokens.back()] = expantion.parentToken;
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
            predeccessor.erase(expandedTokens[i]);
        }
        expandedTokens.resize(newSize);
        // cout<<"New Size due to beam: "<<expandedTokens.size()<<endl;
    }
}

vector<const Arc*> BeamSearch::getPath(shared_ptr<Token> rootToken) {
    vector<const Arc*> arcs = vector<const Arc*>();
    if (activeTokens.size() <= 0) return arcs;

    shared_ptr<Token> currToken = rootToken;
    while (currToken) {
        arcs.push_back(currToken->arc);
        currToken = predeccessor[currToken];
    }
    reverse(begin(arcs), end(arcs));
    return arcs;
}


vector<double> BeamSearch::getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens) {
    if (tokens.size() <= 0) return vector<double>();
    const auto& bestToken = *max_element(begin(tokens), end(tokens), [&](const auto& t1, const auto& t2) {
        return t1->amCost + t1->lmCost < t2->amCost + t2->lmCost;
    });
    double bestCost = bestToken->amCost + bestToken->lmCost;

    vector<double> logProbas(tokens.size(), 0);
    for (uint i = 0; i < tokens.size(); ++i) {
        logProbas[i] = tokens[i]->amCost + tokens[i]->lmCost - bestCost;
    }
    return logProbas;
}

BeamSearch::~BeamSearch() {
}