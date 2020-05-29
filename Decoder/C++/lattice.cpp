#include "lattice.hpp"

Lattice::Lattice(uint latticeBeam, shared_ptr<Token> intialToken) : latticeBeam(latticeBeam), tokenId(1) {
    tokenTopology[intialToken] = Topology();
}

void Lattice::startNewExpantions() {
    expantions.clear();
    arcToToken.clear();
    bestExpantionCost = -numeric_limits<double>::infinity();
}

void Lattice::expand(const shared_ptr<Token>& nextToken, const Arc*& arc, double lmCost, double amCost) {
    auto expantion = Expantion(nextToken, lmCost, amCost);
    bestExpantionCost = max(bestExpantionCost, expantion.lmCost + expantion.amCost);
    expantions[arc].insert(expantion);
};

void Lattice::createExpandedTokens(vector<shared_ptr<Token>>& newTokens, double expantionCostBeam) {
    for (auto& keyVal : expantions) {
        const auto& arc = keyVal.first;
        const auto& expantionsSet = keyVal.second;
        const auto& expantion = *(expantionsSet.begin());
        double lmCost = expantion.lmCost;
        double amCost = expantion.amCost;
        double arcAmCost = expantion.arcAmCost;

        if (arcToToken.find(arc) != arcToToken.end()) {  // this token created before in same time step
            arcToToken[arc]->lmCost = lmCost;
            arcToToken[arc]->amCost = amCost;
            arcToToken[arc]->arcAmCost = arcAmCost;  // this is not neccessary as arcAmCost is constant for same time step
        } else {
            if (lmCost + amCost >= bestExpantionCost - expantionCostBeam) {  // ensure created token cost is within beam
                shared_ptr<Token> newToken = shared_ptr<Token>(new Token(tokenId++, arc, lmCost, amCost, arcAmCost));
                newTokens.push_back(newToken);
                arcToToken[arc] = newToken;
            }
        }
    }
}

void Lattice::finishExpantions(double expantionCostBeam) {
    for (const auto& keyVal : expantions) {
        const auto& arc = keyVal.first;
        const auto& expantionsSet = keyVal.second;

        uint topK = latticeBeam;
        const auto& createdToken = arcToToken[arc];

        for (const auto& expantion : expantionsSet) {
            if (topK-- == 0) break;                                                              // use only best k expantions
            if (expantion.amCost + expantion.lmCost >= bestExpantionCost - expantionCostBeam) {  // ensure kept tokens' cost is within beam
                tokenTopology[createdToken].predecessors.push_back(expantion.parentToken);
                tokenTopology[expantion.parentToken].successorsCount += 1;
            }
        }
    }
    expantions.clear();
    arcToToken.clear();
}

vector<const Arc*> Lattice::getBestPath(shared_ptr<Token> token) {
    vector<const Arc*> arcSeq;
    shared_ptr<Token> currToken = token;
    while (tokenTopology[currToken].predecessors.size()) {
        arcSeq.push_back(currToken->arc);
        currToken = tokenTopology[currToken].predecessors.front();
    }

    reverse(begin(arcSeq), end(arcSeq));
    return arcSeq;
}

void Lattice::removeToken(shared_ptr<Token> token) {
    vector<shared_ptr<Token>> predecessors = tokenTopology[token].predecessors;
    tokenTopology.erase(token);
    for (auto& predecessor : predecessors) {
        auto it = tokenTopology.find(predecessor);
        if (it != tokenTopology.end()) {
            it->second.successorsCount -= 1;
        }
    }
}


void Lattice::writeAsFst(string filename, const vector<shared_ptr<Token>>& finalTokens,const unordered_map<uint, double>& finalStates) {
    ofstream out;
    out.open(filename, fstream::out);
    if (!out.is_open()) {
        throw Exception("Can't open file: " + filename);
    }

    vector<const Arc*> arcs;
    unordered_map<const Arc*, bool> savedArc;
    unordered_map<shared_ptr<Token>, bool> visited;
    vector<shared_ptr<Token>> currTokens(finalTokens.size()), nextTokens;
    set<uint> states;

    // set curr tokens with final tokens
    for (uint i = 0; i < currTokens.size(); i++) {
        currTokens[i] = finalTokens[i];
        visited[currTokens[i]] = true;
    }
    while (currTokens.size()) {
        for (auto& token : currTokens) {
            if (token.get()) {
                if (!savedArc[token->arc]) {
                    if (token->arc->srcState != token->arc->dstState) {
                        arcs.push_back(token->arc);
                        states.insert(token->arc->srcState);
                        states.insert(token->arc->dstState);
                        savedArc[token->arc] = true;
                    }
                }
                for (auto& predecessor : tokenTopology.find(token)->second.predecessors) {
                    if (!visited[predecessor]) {
                        nextTokens.push_back(predecessor);
                        visited[predecessor] = true;
                    }
                }
            }
        }
        currTokens = move(nextTokens);
    }

    // sort arcs based on src states
    sort(begin(arcs), end(arcs), [](const Arc*& arc1, const Arc*& arc2) {
        if (arc1->srcState != arc2->srcState) return arc1->srcState < arc2->srcState;
        return arc1->dstState < arc2->dstState;
    });

    // set states new index
    unordered_map<uint, uint> statesNewIndx, newIndxOfFinalStates;
    uint i = 0;
    for (auto state : states) {
        if (finalStates.find(state) != finalStates.end()) {
            newIndxOfFinalStates[i] = state;
        }
        statesNewIndx[state] = i++;
    }

    // write saved arcs and final stated
    for (i = 0; i < arcs.size(); ++i) {
        // write arc info
        out << statesNewIndx[arcs[i]->srcState] << " " << statesNewIndx[arcs[i]->dstState] << " ";
        out << arcs[i]->inpId << " " << arcs[i]->outId;
        if (arcs[i]->lmCost != 0) {
            out << " " << -arcs[i]->lmCost;
        }
        out << endl;

        // write final state if arc destination is final state
        if (i + 1 == arcs.size() || arcs[i + 1]->srcState != arcs[i]->srcState) {
            auto it = newIndxOfFinalStates.find(statesNewIndx[arcs[i]->srcState] + 1);
            if (it != newIndxOfFinalStates.end()) {
                out << it->first;
                double cost = finalStates.find(it->second)->second;
                if (cost != 0) {
                    out << " " << -cost;
                }
                out << endl;
            }
        }
    }
}