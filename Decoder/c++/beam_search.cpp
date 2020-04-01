#include "beam_search.hpp"
#include <ctime>
#include "fst.hpp"

BeamSearch::BeamSearch(uint beamWidth, double pathAcceptingThreshold) : beamWidth(beamWidth), pathAcceptingThreshold(pathAcceptingThreshold), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
    assert(beamWidth != 0);
}

void BeamSearch::setRootToken(const Arc* arc, double lmCost, double modelCost) {
    expandedTokens.clear();
    activeTokens.clear();
    predeccessor.clear();
    expandNewToken(arc, lmCost, modelCost);
    predeccessor[expandedTokens.front()] = shared_ptr<Token>(NULL);
    moveExpandedToActive();
}

void BeamSearch::keepOnlyBestExpandedTokens() {
    unordered_map<const Arc*, shared_ptr<Token>> arcToBestToken;
    for (const auto& token : expandedTokens) {
        double tokenCost = token->modelCost + token->lmCost;
        auto iter = arcToBestToken.find(token->arc);
        if (iter == arcToBestToken.end()) {
            arcToBestToken[token->arc] = token;
        } else {
            if (tokenCost > iter->second->modelCost + iter->second->lmCost) {
                iter->second = token;
            }
        }
    }

    expandedTokens.erase(remove_if(begin(expandedTokens), end(expandedTokens), [&](const auto& t) { return arcToBestToken[t->arc] != t; }),
                         end(expandedTokens));
}

void BeamSearch::doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, bool useSelfLoops) {
    unordered_map<const Arc*, Expantion> expantions;  // map expanded node to parent node and expantion cost
    vector<double> logProbas = getNormalizeTokensLogProba(activeTokens);
    // vector<int> activationsRank(activations.size());
    // iota(begin(activationsRank), end(activationsRank), 0);
    // sort(begin(activationsRank), end(activationsRank), [&](auto a, auto b) {
    //     return activations[a] > activations[b];
    // });

    // clock_t begin_time = clock();
    // cout << "Time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << endl;
    if (useSelfLoops) {
        for (uint i = 0; i < activeTokens.size(); ++i) {
            const auto& token = activeTokens[i];
            auto iter = inpLabelsToIndx.find(token->arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            double modelCost = activations[iter->second];
            double specialCost = (token->arc->srcState == token->arc->dstState) ? log(2.) : 0;
            double expantionCost = logProbas[i];
            expantions[token->arc] = Expantion(token, 0., modelCost, expantionCost);
        }
    }

    for (uint i = 0; i < activeTokens.size(); ++i) {
        const auto& token = activeTokens[i];
        for (const Arc* arc : graph[token->arc->dstState]) {
            double lmCost = arc->lmCost;
            double transCost = arc->transCost;
            double modelCost = 0;

            auto iter = inpLabelsToIndx.find(arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            modelCost = activations[iter->second];

            //Expand the frontier and add predecessors
            double expantionCost = logProbas[i] + lmCost;
            if (exp(expantionCost) > pathAcceptingThreshold) {
                bool isNewArc = expantions.find(arc) == expantions.end();
                if (isNewArc) {
                    expantions[arc] = Expantion(token, lmCost, modelCost, expantionCost);
                } else {
                    double oldCost = expantions[arc].expantionCost;
                    if (expantionCost > oldCost) {
                        expantions[arc] = Expantion(token, lmCost, modelCost, expantionCost);
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

        double modelCost = expantion.parentToken->modelCost + expantion.modelCost;
        double lmCost = expantion.parentToken->lmCost + expantion.lmCost;

        expandNewToken(arc, lmCost, modelCost);
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
            return (a->lmCost + a->modelCost) > (b->lmCost + b->modelCost);
        });
        expandedTokens.resize(beamWidth);
    }
}

vector<const Arc*> BeamSearch::getBestPath(const vector<vector<const Arc*>>& graph, Token& finalToken) {
    vector<const Arc*> arcs = vector<const Arc*>();
    if (activeTokens.size() <= 0) return arcs;

    const auto& bestToken = *max_element(begin(activeTokens), end(activeTokens), [](const auto& t1, const auto& t2) {
        return t1->modelCost + t1->lmCost < t2->modelCost + t2->lmCost;
    });

    finalToken = *(bestToken);
    shared_ptr<Token> currToken = bestToken, parentToken;
    while (currToken) {
        arcs.push_back(currToken->arc);
        parentToken = predeccessor[currToken];
        currToken = parentToken;
    }
    reverse(begin(arcs), end(arcs));
    return arcs;
}

vector<double> BeamSearch::getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens) {
    if (tokens.size() <= 0) return vector<double>();
    const auto& bestToken = *max_element(begin(tokens), end(tokens), [](const auto& t1, const auto& t2) {
        return t1->modelCost + t1->lmCost < t2->modelCost + t2->lmCost;
    });
    double bestCost = bestToken->modelCost + bestToken->lmCost;

    vector<double> logProbas(tokens.size(), 0);
    for (uint i = 0; i < tokens.size(); ++i) {
        logProbas[i] = tokens[i]->lmCost + tokens[i]->modelCost - bestCost;
    }
    return move(logProbas);
}

BeamSearch::~BeamSearch() {
}