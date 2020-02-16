#include "beam_search.hpp"
#include "fst.hpp"

BeamSearch::BeamSearch(uint beamWidth) : beamWidth(beamWidth), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
    assert(beamWidth != 0);
}

void BeamSearch::setRootToken(const Arc* arc, double lmScore, double modelScore) {
    expandedTokens.clear();
    activeTokens.clear();
    expandNewToken(arc, lmScore, modelScore);
    predeccessor[expandedTokens.front()] = shared_ptr<Token>(nullptr);
    moveExpandedToActive();
}

void BeamSearch::keepOnlyBestExpandedTokens(int start) {
    unordered_map<const Arc*, shared_ptr<Token>> arcToBestToken;
    for (int i = start; i < expandedTokens.size(); ++i) {
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

    int count = moveRelevantFisrt<shared_ptr<Token>>(expandedTokens,
                                                     [&](const auto& t) { return arcToBestToken[t->arc] == t; }, start);
    expandedTokens.resize(count);
}

void BeamSearch::doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, bool useSelfLoops) {
    unordered_map<const Arc*, Expantion> expantions;  // map expanded node to parent node and expantion cost
    vector<double> probas = getNormalizeTokensProba(activeTokens);

    int tokIndx = 0;
    if (useSelfLoops) {
        for (const auto& tokenPtr : activeTokens) {
            auto iter = inpLabelsToIndx.find(tokenPtr->arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            double modelScore = activations[iter->second];
            expantions[tokenPtr->arc] = Expantion(tokIndx, 0., modelScore);
            tokIndx++;
        }
    }

    tokIndx = 0;
    for (const auto& tokenPtr : activeTokens) {
        for (const Arc* arc : graph[tokenPtr->arc->dstState]) {
            double lmScore = exp(arc->cost);
            double modelScore = 0;

            auto iter = inpLabelsToIndx.find(arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            modelScore = activations[iter->second];

            //Expand the frontier and add predecessors
            double relativeExpantionScore = probas[tokIndx] * lmScore;

            if (relativeExpantionScore > 0) {
                bool isNewArc = expantions.find(arc) == expantions.end();

                if (isNewArc) {
                    expantions[arc] = Expantion(tokIndx, log(lmScore), modelScore);
                } else {
                    double oldScore = probas[expantions[arc].parentTokenIndx] * lmScore;
                    if (relativeExpantionScore > oldScore) {
                        expantions[arc] = Expantion(tokIndx, log(lmScore), modelScore);
                    }
                }
            }
        }
        ++tokIndx;
    }

    // create new tokens
    createExpandedTokens(expantions);
}

void BeamSearch::createExpandedTokens(const unordered_map<const Arc*, Expantion>& expantions) {
    for (const auto& keyVal : expantions) {
        const auto& expantion = keyVal.second;
        const Arc* arc = keyVal.first;

        auto parentToken = activeTokens[expantion.parentTokenIndx];
        double modelScore = parentToken->modelScore + expantion.modelScore, lmScore = parentToken->lmScore + expantion.lmScore;

        expandNewToken(arc, lmScore, modelScore);
        predeccessor[expandedTokens.back()] = parentToken;
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
    int relevantCount = moveRelevantFisrt<shared_ptr<Token>>(activeTokens, [&](const auto& t) {
        return finalStates.find(t->arc->dstState) != finalStates.end();  // it's a final state
    });

    for (int i = 0; i < relevantCount; ++i) {
        activeTokens[i]->lmScore += finalStates.find(activeTokens[i]->arc->dstState)->second;  // add final state cost
    }

    activeTokens.resize(relevantCount);  // remove non final states
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

void Token::print(ostream& out) const {
    out << "Token id: " << tokId << " inputlabel: " << (arc->inpLabel) << "outputlabel: " << (arc->outLabel) << " lmScore: " << lmScore << " modelScore: " << modelScore << " Joint score:" << modelScore + lmScore << endl;
}