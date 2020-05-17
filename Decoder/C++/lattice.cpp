#include "lattice.hpp"

Lattice::Lattice(uint latticeBeam) : latticeBeam(latticeBeam), tokenId(1) {
}

void Lattice::startNewExpantions() {
    expantions.clear();
    arcToToken.clear();
    bestExpantionCost = -numeric_limits<double>::infinity();
}

void Lattice::expand(const shared_ptr<Token>& parent, const Arc*& arc, double lmCost, double amCost) {
    expantions[arc].insert(Expantion(parent, lmCost, amCost));
};

void Lattice::createExpandedTokens(vector<shared_ptr<Token>>& newTokens, double expantionCostBeam) {
    for (auto& keyVal : expantions) {
        const auto& arc = keyVal.first;
        auto& expantionsSet = keyVal.second;
        double lmCost = expantionsSet.begin()->lmCost;
        double amCost = expantionsSet.begin()->amCost;

        if (arcToToken.find(arc) != arcToToken.end()) {  // this token created before in same time step
            arcToToken[arc]->lmCost = lmCost;
            arcToToken[arc]->amCost = amCost;
        } else {
            if (lmCost + amCost <= bestExpantionCost - expantionCostBeam) { // ensure created token cost is within beam
                shared_ptr<Token> newToken = shared_ptr<Token>(new Token(tokenId++, arc, lmCost, amCost));
                newTokens.push_back(newToken);
                arcToToken[arc] = newToken;
            }
        }
    }
}

void Lattice::finishExpantions() {
    for (const auto& keyVal : expantions) {
        const auto& arc = keyVal.first;
        const auto& expantionsSet = keyVal.second;

        uint topK = latticeBeam;
        const auto& createdToken = arcToToken[arc];

        if (tokenToplogy.find(createdToken) != tokenToplogy.end()) {
            assert(tokenToplogy[createdToken].successorsCount);
        }
        for (const auto& expantion : expantionsSet) {
            if (topK-- == 0) break;  // use only best k expantions
            tokenToplogy[createdToken].predecessors.push_back(expantion.parentToken);
            tokenToplogy[expantion.parentToken].successorsCount += 1;
        }
    }
    expantions.clear();
    arcToToken.clear();
}

vector<const Arc*> Lattice::getBestPath(shared_ptr<Token> token) {
    vector<const Arc*> arcSeq;
    shared_ptr<Token> currToken = token;
    while (currToken) {
        arcSeq.push_back(currToken->arc);
        currToken = tokenToplogy[currToken].predecessors.front();
    }

    reverse(begin(arcSeq), end(arcSeq));
    return arcSeq;
}

void Lattice::removeToken(shared_ptr<Token> token) {
    vector<shared_ptr<Token>> predecessors = tokenToplogy[token].predecessors;
    tokenToplogy.erase(token);
    for (auto& predecessor : predecessors) {
        auto it = tokenToplogy.find(predecessor);
        if (it != tokenToplogy.end()) {
            it->second.successorsCount -= 1;
        }
    }
}

void Lattice::writeAsFst(string filename, const vector<shared_ptr<Token>>& finalTokens, const unordered_map<uint, double>& finalStates) const {
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
                for (auto& predecessor : tokenToplogy.find(token)->second.predecessors) {
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

// void Lattice::printAllHyps(shared_ptr<Token> token, const unordered_map<uint, string>& outSymTable) {
//     static vector<const Arc*> arcSeq;
//     if (!token) {
//         vector<const Arc*> reversedPath(rbegin(arcSeq), rend(arcSeq));
//         for (int i = 0; i < reversedPath.size(); ++i) {
//             if (i && reversedPath[i]->dstState == reversedPath[i - 1]->dstState) {
//                 continue;
//             }
//             string outSym = outSymTable.find(reversedPath[i]->outId)->second;
//             if (outSym !="<eps>" && outSym !="</s>" && outSym !="<s>") {
//                 cout << outSym << " ";
//             }
//         }
//         cout << endl;
//     } else {
//         arcSeq.push_back(token->arc);
//         for (const auto& predecessor : tokenToplogy[token].predecessors) {
//             printAllHyps(predecessor, outSymTable);
//         }
//         arcSeq.pop_back();
//     }
// }

// void Lattice::expandLattice(const vector<shared_ptr<Token>>& newTokens, const unordered_map<uint, uint>& inpIdsToActivationsIndx, const vector<double>& activations) {
//     vector<const Arc*> newArcs = vector<const Arc*>(newTokens.size());
//     transform(begin(newTokens), end(newTokens), begin(newArcs), [](const auto& token) { return token->arc; });

//     // sort current leaves on dst state
//     sort(begin(currLeaves), end(currLeaves), [](const auto& l1, const auto& l2) {
//         return l1->arc->dstState < l2->arc->dstState;
//     });

//     // create next leaves
//     vector<LatticeNode*> nextLeaves(newArcs.size());
//     for (size_t i = 0; i < nextLeaves.size(); i++) {
//         nextLeaves[i] = new LatticeNode(newArcs[i]);
//     }

//     // sort next leaves on src state
//     sort(begin(nextLeaves), end(nextLeaves), [](const auto& l1, const auto& l2) {
//         return l1->arc->srcState < l2->arc->srcState;
//     });

//     // connect currLeaves to next leaves
//     for (size_t curri = 0, nexti = 0; curri < nextLeaves.size(); curri++) {
//         for (; nexti < nextLeaves.size() && nextLeaves[nexti]->arc->srcState == currLeaves[curri]->arc->dstState; ++nexti) {
//             currLeaves[curri]->predecessors.push_back(unique_ptr<LatticeNode>(nextLeaves[nexti]));
//             currLeaves[curri]->predecessors.back()->parent = currLeaves[curri];
//             nextLeaves[nexti]->amCost = activations[inpIdsToActivationsIndx.find(nextLeaves[nexti]->arc->inpId)->second];
//             nextLeaves[nexti]->cumulativeCost = nextLeaves[nexti]->amCost + nextLeaves[nexti]->arc->lmCost + currLeaves[curri]->cumulativeCost;
//         }

//         // remove leaf if it has no predecessors
//         pruneChildLess(currLeaves[curri]);
//     }

//     // move next leaves to current leaves
//     currLeaves = move(nextLeaves);
// }
