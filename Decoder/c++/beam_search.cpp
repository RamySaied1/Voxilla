#include "beam_search.hpp"
#include <ctime>
#include "fst.hpp"

BeamSearch::BeamSearch(uint beamWidth, double pathAcceptingThreshold) : beamWidth(beamWidth), pathAcceptingThreshold(pathAcceptingThreshold), activeTokens(vector<shared_ptr<Token>>()), predeccessor(unordered_map<shared_ptr<Token>, shared_ptr<Token>>()) {
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
    for (const auto& token : expandedTokens) {
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
            double modelScore = activations[iter->second];
            double lmScore = 0.;
            double specialScore = (token->arc->srcState == token->arc->dstState) ? log(2.) : 0;
            double expantionScore = logProbas[i] + lmScore;
            expantions[token->arc] = Expantion(token, lmScore, modelScore, expantionScore);
        }
    }

    for (uint i = 0; i < activeTokens.size(); ++i) {
        const auto& token = activeTokens[i];
        for (const Arc* arc : graph[token->arc->dstState]) {
            double lmScore = arc->cost;
            double modelScore = 0;

            auto iter = inpLabelsToIndx.find(arc->inpLabel);
            if (iter == inpLabelsToIndx.end()) continue;
            modelScore = activations[iter->second];

            //Expand the frontier and add predecessors
            double expantionScore = logProbas[i] + lmScore;
            if (exp(expantionScore) > pathAcceptingThreshold) {
                bool isNewArc = expantions.find(arc) == expantions.end();
                if (isNewArc) {
                    expantions[arc] = Expantion(token, lmScore, modelScore, expantionScore);
                } else {
                    double oldScore = expantions[arc].expantionScore;
                    if (expantionScore > oldScore) {
                        expantions[arc] = Expantion(token, lmScore, modelScore, expantionScore);
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
        } while (parentToken && currToken->arc == parentToken->arc);  // ignoring self loop
    }
    reverse(begin(arcs), end(arcs));
    return arcs;
}

vector<double> BeamSearch::getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens) {
    if (tokens.size() <= 0) return vector<double>();
    const auto& bestToken = *max_element(begin(tokens), end(tokens), [](const auto& t1, const auto& t2) {
        return t1->modelScore + t1->lmScore < t2->modelScore + t2->lmScore;
    });
    double bestScore = bestToken->modelScore + bestToken->lmScore;

    vector<double> logProbas(tokens.size(), 0);
    for (uint i = 0; i < tokens.size(); ++i) {
        logProbas[i] = tokens[i]->lmScore + tokens[i]->modelScore - bestScore;
    }
    return move(logProbas);
}

BeamSearch::~BeamSearch() {
}