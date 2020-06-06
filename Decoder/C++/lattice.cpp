#include "lattice.hpp"

Lattice::Lattice(uint latticeBeam, shared_ptr<Token> intialToken) : latticeBeam(latticeBeam), tokenId(1) {
    tokenTopology[intialToken] = Topology();
}

void Lattice::startNewExpantions() {
    expantions.clear();
    arcToToken.clear();
    bestExpantionCost = -numeric_limits<double>::infinity();
}

void Lattice::expand(const shared_ptr<Token>& parentToken, const Arc*& arc, double lmCost, double amCost) {
    auto expantion = Expantion(parentToken, lmCost, amCost);
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

        bool isNewArcExpan = arcToToken.find(arc) == arcToToken.end();
        bool isExpanCostWithinBeam = lmCost + amCost > bestExpantionCost - expantionCostBeam;
        bool isBetterExpanForOldArc = !isNewArcExpan && lmCost + amCost > arcToToken[arc]->lmCost + arcToToken[arc]->amCost;

        if ((isNewArcExpan && isExpanCostWithinBeam) || isBetterExpanForOldArc) {
            shared_ptr<Token> newToken = shared_ptr<Token>(new Token(tokenId++, arc, lmCost, amCost, arcAmCost));
            newTokens.push_back(newToken);
            arcToToken[arc] = newToken;
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
            if (topK-- == 0) break;                                                             // use only best k expantions
            if (expantion.amCost + expantion.lmCost > bestExpantionCost - expantionCostBeam) {  // ensure kept tokens' cost is within beam
                tokenTopology[createdToken].predecessors.push_back(expantion.parentToken);
                tokenTopology[expantion.parentToken].successorsCount += 1;
            } else {
                break;
            }
        }
    }
    startNewExpantions();  // save memory
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

void Lattice::latticToFst(const vector<shared_ptr<Token>>& finalTokens, const unordered_map<uint, double>& finalStates, fst::StdVectorFst* fst) {
    vector<Arc> arcs;
    unordered_map<const Arc*, bool> arcCopied;
    unordered_map<shared_ptr<Token>, bool> visited;
    vector<shared_ptr<Token>> currTokens(finalTokens.size()), nextTokens;

    // set curr tokens with final tokens
    for (uint i = 0; i < currTokens.size(); i++) {
        currTokens[i] = finalTokens[i];
        visited[currTokens[i]] = true;
    }
    while (currTokens.size()) {
        for (auto& token : currTokens) {
            if (token.get()) {
                if (!arcCopied[token->arc]) {
                    if (token->arc->srcState != token->arc->dstState) {
                        arcs.push_back(*(token->arc));
                        arcCopied[token->arc] = true;
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
    sort(begin(arcs), end(arcs), [](const Arc& arc1, const Arc& arc2) {
        if (arc1.srcState != arc2.srcState) return arc1.srcState < arc2.srcState;
        return arc1.dstState < arc2.dstState;
    });

    // get new states indicies
    uint i = 0;
    unordered_map<uint, uint> statesNewIndx;
    map<uint, double> myFinalStates;
    for (auto& arc : arcs) {
        if (statesNewIndx.find(arc.srcState) == statesNewIndx.end()) {
            statesNewIndx[arc.srcState] = i++;
        }
    }
    // transfrom states to new indicies and add myfinalstates
    for (auto& arc : arcs) {
        arc.srcState = statesNewIndx[arc.srcState];
        auto it = finalStates.find(arc.dstState);
        if (statesNewIndx.find(arc.dstState) == statesNewIndx.end()) {
            statesNewIndx[arc.dstState] = i++;
            assert(it != finalStates.end());
        }
        if (it != finalStates.end()) {
            myFinalStates[statesNewIndx[arc.dstState]] = it->second;
        }
        arc.dstState = statesNewIndx[arc.dstState];
    }

    fst->AddState();
    fst->SetStart(0);
    uint lastCreatedState = 0;
    // create fst arcs
    for (i = 0; i < arcs.size(); ++i) {
        if (arcs[i].srcState != lastCreatedState) {
            fst->AddState();
            lastCreatedState = arcs[i].srcState;
        }

        double cost = -arcs[i].lmCost;
        fst->AddArc(arcs[i].srcState, fst::StdArc(arcs[i].inpId, arcs[i].outId, cost, arcs[i].dstState));
    }

    // set final states
    for (const auto& keyVal : myFinalStates) {
        uint state = keyVal.first;
        double cost = -keyVal.second;
        if (state > lastCreatedState) {
            fst->AddState();
            lastCreatedState = state;
        }
        fst->SetFinal(state, cost);
    }
}