#include "beam_search.hpp"
#include <ctime>
#include "fst.hpp"

BeamSearch::BeamSearch(uint beamWidth, double pathAcceptingThreshold) : beamWidth(beamWidth), pathAcceptingThreshold(pathAcceptingThreshold), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
    assert(beamWidth != 0);
}

void BeamSearch::setRootToken(const Arc* arc, double lmCost, double amCost, double hmmCost) {
    expandedTokens.clear();
    activeTokens.clear();
    predeccessor.clear();
    expandNewToken(arc, lmCost, amCost,hmmCost);
    predeccessor[expandedTokens.front()] = shared_ptr<Token>(NULL);
    moveExpandedToActive();
}

void BeamSearch::keepOnlyBestExpandedTokens(function<double(double, double, double)> combineAmLmHmmCosts) {
    unordered_map<const Arc*, shared_ptr<Token>> arcToBestToken;
    for (const auto& token : expandedTokens) {
        double tokenCost = combineAmLmHmmCosts(token->amCost, token->lmCost, token->hmmCost);
        auto iter = arcToBestToken.find(token->arc);
        if (iter == arcToBestToken.end()) {
            arcToBestToken[token->arc] = token;
        } else {
            if (tokenCost > combineAmLmHmmCosts(iter->second->amCost, iter->second->lmCost, iter->second->hmmCost)) {
                iter->second = token;
            }
        }
    }

    expandedTokens.erase(remove_if(begin(expandedTokens), end(expandedTokens), [&](const auto& t) { return arcToBestToken[t->arc] != t; }),
                         end(expandedTokens));
}

void BeamSearch::doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, function<double(double, double, double)> combineAmLmHmmCosts, bool useSelfLoops) {
    unordered_map<const Arc*, Expantion> expantions;  // map expanded node to parent node and expantion cost
    vector<double> logProbas = getNormalizeTokensLogProba(activeTokens, combineAmLmHmmCosts);
    // vector<int> activationsRank(activations.size());
    // iota(begin(activationsRank), end(activationsRank), 0);
    // sort(begin(activationsRank), end(activationsRank), [&](auto a, auto b) {
    //     return activations[a] > activations[b];
    // });

    // clock_t begin_time = clock();
    // cout << "Time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << endl;
    // if (useSelfLoops) {
    //     for (uint i = 0; i < activeTokens.size(); ++i) {
    //         const auto& token = activeTokens[i];
    //         auto iter = inpLabelsToIndx.find(token->arc->inpLabel);
    //         if (iter == inpLabelsToIndx.end()) continue;
    //         // double lmCost = token->arc->lmCost;
    //         double amCost = activations[iter->second];
    //         double specialCost = (token->arc->srcState == token->arc->dstState) ? log(2.) : 0;
    //         double expantionCost = logProbas[i];
    //         expantions[token->arc] = Expantion(token, 0., amCost, expantionCost);
    //     }
    // }

    for (uint i = 0; i < activeTokens.size(); ++i) {
        const auto& token = activeTokens[i];
        for (const Arc* arc : graph[token->arc->dstState]) {
            auto iter = inpLabelsToIndx.find(arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            double lmCost = arc->lmCost;
            double hmmCost = arc->transCost;
            double amCost = activations[iter->second];

            //Expand the frontier and add predecessors
            double expantionCost = logProbas[i] + combineAmLmHmmCosts(amCost, lmCost, hmmCost);
            if (exp(expantionCost) > pathAcceptingThreshold) {
                bool isNewArc = expantions.find(arc) == expantions.end();
                if (isNewArc) {
                    expantions[arc] = Expantion(token, lmCost, amCost, hmmCost, expantionCost);
                } else {
                    double oldCost = expantions[arc].expantionCost;
                    if (expantionCost > oldCost) {
                        expantions[arc] = Expantion(token, lmCost, amCost, hmmCost, expantionCost);
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
        double hmmCost = expantion.parentToken->hmmCost + expantion.hmmCost;

        expandNewToken(arc, lmCost, amCost, hmmCost);
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

void BeamSearch::beamPrune(function<double(double, double, double)> combineAmLmHmmCosts) {
    if (expandedTokens.size() > beamWidth) {
        sort(begin(expandedTokens), end(expandedTokens), [&](const shared_ptr<Token>& a, const shared_ptr<Token>& b) {
            return combineAmLmHmmCosts(a->amCost, a->lmCost, a->hmmCost) > combineAmLmHmmCosts(b->amCost, b->lmCost, b->hmmCost);
        });
        expandedTokens.resize(beamWidth);
    }
}

vector<const Arc*> BeamSearch::getBestPath(const vector<vector<const Arc*>>& graph, function<double(double, double, double)> combineAmLmHmmCosts, Token& finalToken) {
    vector<const Arc*> arcs = vector<const Arc*>();
    if (activeTokens.size() <= 0) return arcs;

    const auto& bestToken = *max_element(begin(activeTokens), end(activeTokens), [&](const auto& t1, const auto& t2) {
        return combineAmLmHmmCosts(t1->amCost, t1->lmCost, t1->hmmCost) < combineAmLmHmmCosts(t2->amCost, t2->lmCost, t2->hmmCost);
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

vector<double> BeamSearch::getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens, function<double(double, double, double)> combineAmLmHmmCosts) {
    if (tokens.size() <= 0) return vector<double>();
    const auto& bestToken = *max_element(begin(tokens), end(tokens), [&](const auto& t1, const auto& t2) {
        return combineAmLmHmmCosts(t1->amCost, t1->lmCost, t1->hmmCost) < combineAmLmHmmCosts(t2->amCost, t2->lmCost, t2->hmmCost);
    });
    double bestCost = combineAmLmHmmCosts(bestToken->amCost, bestToken->lmCost, bestToken->hmmCost);

    vector<double> logProbas(tokens.size(), 0);
    for (uint i = 0; i < tokens.size(); ++i) {
        logProbas[i] = combineAmLmHmmCosts(tokens[i]->amCost, tokens[i]->lmCost, tokens[i]->hmmCost) - bestCost;
    }
    return move(logProbas);
}

BeamSearch::~BeamSearch() {
}