#include "beam_search.hpp"
#include "fst.hpp"

BeamSearch::BeamSearch(uint beamWidth) : beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
    assert(beamWidth != 0);
}

void BeamSearch::setRootToken(const Arc* arc, double lmScore, double modelScore) {
    expandedTokens.clear();
    activeTokens.clear();
    predeccessor.clear();
    expandNewToken(arc, lmScore, modelScore);
    predeccessor[expandedTokens.front()] = shared_ptr<Token>(NULL);
    moveExpandedToActive();
}

void BeamSearch::keepOnlyBestExpandedTokens() {
    unordered_map<const Arc*, shared_ptr<Token>> arcToBestToken;
    for (int i = 0; i < expandedTokens.size(); ++i) {
        const auto& token = expandedTokens[i];
        double tokenScore = token->modelScore + token->lmScore;
        auto iter = arcToBestToken.find(token->arc);
        if (iter == arcToBestToken.end()) {
            arcToBestToken[token->arc] = token;
        } else {
            if (tokenScore > iter->second->modelScore + iter->second->lmScore) {
                iter->second = token;
            }
        }
    }

    expandedTokens.erase(remove_if(begin(expandedTokens), end(expandedTokens), [&](const auto& t) { return arcToBestToken[t->arc] != t; }),
                         end(expandedTokens));
}

void BeamSearch::doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, bool useSelfLoops) {
    unordered_map<const Arc*, Expantion> expantions;  // map expanded node to parent node and expantion cost
    // vector<double> probas = getNormalizeTokensProba(activeTokens);

    if (useSelfLoops) {
        for (const auto& tokenPtr : activeTokens) {
            if (tokenPtr->arc->srcState == tokenPtr->arc->dstState) continue;
            auto iter = inpLabelsToIndx.find(tokenPtr->arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            double modelScore = activations[iter->second];
            double lmScore = 0.;
            expantions[tokenPtr->arc] = Expantion(tokenPtr, lmScore, modelScore);
        }
    }

    for (const auto& tokenPtr : activeTokens) {
        for (const Arc* arc : graph[tokenPtr->arc->dstState]) {
            double lmScore = arc->cost;
            double modelScore = 0;

            auto iter = inpLabelsToIndx.find(arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            modelScore = activations[iter->second];

            //Expand the frontier and add predecessors
            double expantionScore = tokenPtr->lmScore + tokenPtr->modelScore + lmScore;
            if (exp(expantionScore) > 0) {
                bool isNewArc = expantions.find(arc) == expantions.end();
                if (isNewArc) {
                    expantions[arc] = Expantion(tokenPtr, lmScore, modelScore);
                } else {
                    double oldScore = expantions[arc].parentToken->lmScore + expantions[arc].parentToken->modelScore + expantions[arc].lmScore;
                    if (expantionScore > oldScore) {
                        expantions[arc] = Expantion(tokenPtr, lmScore, modelScore);
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

        double modelScore = expantion.parentToken->modelScore + expantion.modelScore;
        double lmScore = expantion.parentToken->lmScore + expantion.lmScore;

        expandNewToken(arc, lmScore, modelScore);
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
    if (expandedTokens.size() > beamWidth) {
        sort(begin(expandedTokens), end(expandedTokens), [](const shared_ptr<Token>& a, const shared_ptr<Token>& b) {
            return (a->lmScore + a->modelScore) > (b->lmScore + b->modelScore);
        });
        expandedTokens.resize(beamWidth);
    }
}

vector<const Arc*> BeamSearch::getBestPath(const vector<vector<const Arc*>>& graph, Token& finalToken) {
    vector<const Arc*> arcs = vector<const Arc*>();
    if (activeTokens.size() <= 0) return arcs;

    const auto& bestToken = *max_element(begin(activeTokens), end(activeTokens), [](const auto& t1, const auto& t2) {
        return t1->modelScore + t1->lmScore < t2->modelScore + t2->lmScore;
    });

    finalToken = *(bestToken);
    shared_ptr<Token> currToken = bestToken, parentToken = predeccessor[bestToken];
    while (parentToken) {
        arcs.push_back(currToken->arc);
        do {
            currToken = parentToken;
            parentToken = predeccessor[currToken];
        } while (parentToken && currToken->arc->dstState == parentToken->arc->dstState);  // ignoring self loop
    }
    reverse(begin(arcs), end(arcs));
    return arcs;
}

void BeamSearch::applyFinalState(const unordered_map<uint, double>& finalStates) {
    auto iend = remove_if(begin(activeTokens), end(activeTokens), [&](const auto& t) {
        return finalStates.find(t->arc->dstState) == finalStates.end();  // remove if it's not a final state
    });

    for (auto i = begin(activeTokens); i != iend; ++i) {
        (*i)->lmScore += finalStates.find((*i)->arc->dstState)->second;  // add final state cost
    }

    activeTokens.erase(iend, end(activeTokens));  // remove non final states
}

vector<double> BeamSearch::getNormalizeTokensProba(const vector<shared_ptr<Token>>& tokens) {
    if (tokens.size() <= 0) return vector<double>();
    const auto& bestToken = *max_element(begin(tokens), end(tokens), [](const auto& t1, const auto& t2) {
        return t1->modelScore + t1->lmScore < t2->modelScore + t2->lmScore;
    });
    double bestScore = bestToken->modelScore + bestToken->lmScore;

    vector<double> probas(tokens.size(), 0);
    for (int i = 0; i < tokens.size(); ++i) {
        probas[i] = exp(tokens[i]->lmScore + tokens[i]->modelScore - bestScore);
    }
    return move(probas);
}

BeamSearch::~BeamSearch() {
}