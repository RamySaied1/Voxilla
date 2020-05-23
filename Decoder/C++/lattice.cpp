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
            if (topK-- == 0) break;                                                                    // use only best k expantions
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

void Lattice::createFstArc(set<Arc, ArcComparator>& arcs, const shared_ptr<Token>& token, const shared_ptr<Token>& prevToken, bool isPrevTokenConnectWithMe, double accumCost, unordered_map<const Token*, uint>& tokenToSrcState, unordered_map<const Token*, uint>& tokenToDstState, uint& id) {
    bool onlyHasSelfLoopPredecessor = true;
    uint mySrcState, dstState;
    auto& predecessors = tokenTopology[token].predecessors;

    if (tokenToDstState.find(token.get()) != tokenToDstState.end() && isPrevTokenConnectWithMe) {
        tokenToSrcState[prevToken.get()] = tokenToDstState[token.get()];
        return;
    }

    // intial state base case
    if (predecessors.size() == 0) {
        mySrcState = tokenToSrcState[token.get()] = 0;  // intial state always know it src
        onlyHasSelfLoopPredecessor = false;             // i don't has predecssors at all
    } else {
        // Make sure to iterate over self-loop last
        vector<uint> indicies(predecessors.size());
        iota(begin(indicies), end(indicies), 0);
        for (uint i = 0; i < predecessors.size(); ++i) {
            if (predecessors[indicies[i]]->arc == token->arc) {
                swap(indicies[i], indicies[0]);
                break;
            }
        }
        for (uint i = 0; i < predecessors.size(); ++i) {
            const shared_ptr<Token>& nextToken = predecessors[indicies[i]];
            if (nextToken->arc == token->arc) {                                                                                       // selfloop
                createFstArc(arcs, nextToken, prevToken, false, accumCost + token->arcAmCost, tokenToSrcState, tokenToDstState, id);  //assuming the self loop token at begging of each predecessor vector
            } else {
                createFstArc(arcs, nextToken, token, true, 0, tokenToSrcState, tokenToDstState, id);
                onlyHasSelfLoopPredecessor = false;
            }
        }
    }
    if (tokenToSrcState.find(prevToken.get()) == tokenToSrcState.end()) {
        if (tokenToDstState.find(token.get()) == tokenToDstState.end()) {
            dstState = id++;
        } else {
            dstState = tokenToDstState[token.get()];
        }
        tokenToSrcState[prevToken.get()] = dstState;
    } else {
        dstState = tokenToSrcState[prevToken.get()];
    }

    if (isPrevTokenConnectWithMe) {
        tokenToDstState[token.get()] = dstState;
    }

    if (!onlyHasSelfLoopPredecessor) {
        mySrcState = tokenToSrcState[token.get()];
        double arcCost = accumCost + token->arcAmCost + token->arc->lmCost;
        Arc newArc{mySrcState, dstState, token->arc->inpId, token->arc->outId, arcCost};
        // arc comparator doesn't consider the lm cost
        auto it = arcs.find(newArc);
        if (it == arcs.end()) {
            arcs.insert(newArc);
        } else {
            if (arcCost > it->lmCost) {  // cost is negative so take if maximum
                arcs.erase(*it);
                arcs.insert(newArc);
            }
        }
    }
}

void Lattice::constructFst(const vector<shared_ptr<Token>>& finalTokens, const vector<double>& finalTokensCost, vector<Arc>& arcs, unordered_map<uint, double>& finalStates) {
    uint id = 1;
    unordered_map<const Token*, uint> tokenToSrcState, tokenToDstState;
    vector<shared_ptr<Token>> dummyTokens(finalTokens.size());
    set<Arc, ArcComparator> arcsSet;
    for (uint i = 0; i < finalTokens.size(); ++i) {
        dummyTokens[i] = shared_ptr<Token>(new Token());
        createFstArc(arcsSet, finalTokens[i], dummyTokens[i], true, 0, tokenToSrcState, tokenToDstState, id);
        uint finalState = tokenToSrcState[dummyTokens[i].get()];
        finalStates[finalState] = finalTokensCost[i];
    }
    arcs = vector<Arc>(begin(arcsSet), end(arcsSet));
    // vector<shared_ptr<Token>> currTokens = finalTokens, nextTokens;
    // vector<vector<pair<uint, const Arc*>>> timeline;
    // unordered_map<const Arc*, double> finalArcs;
    // while (currTokens.size()) {
    //     timeline.push_back(vector<pair<uint, const Arc*>>());
    //     unordered_map<Token*, bool> alreadyInserted;
    //     if (timeline.size() == 1) {  // first time
    //         for (uint i = 0; i < finalTokens.size(); ++i) {
    //             if (!alreadyInserted[finalTokens[i].get()]) {
    //                 finalArcs[finalTokens[i]->arc] = finalTokensCost[i];
    //                 alreadyInserted[finalTokens[i].get()] = true;
    //             }
    //         }
    //     }
    //     for (auto& token : currTokens) {
    //         auto& predecessors = tokenTopology.find(token)->second.predecessors;
    //         if (!predecessors.front().get()) {  // intial token discarded
    //             timeline.pop_back();
    //             break;
    //         }
    //         timeline.back().push_back({predecessors.front()->arc->dstState, token->arc});
    //         for (auto& predecessor : predecessors) {
    //             if (!alreadyInserted[predecessor.get()]) {
    //                 nextTokens.push_back(predecessor);
    //                 alreadyInserted[predecessor.get()] = true;
    //             }
    //         }
    //     }
    //     currTokens = move(nextTokens);
    // }

    // id = 1;
    // unordered_map<uint, uint> currTimedStateToState, prevTimedStateToState = {{(uint)0, (uint)0}};
    // for (auto it = rbegin(timeline); it != rend(timeline); ++it) {
    //     for (auto stateWithArc : *it) {
    //         uint srcState, dstState, inId, outId;
    //         double lmCost;
    //         bool selfLoop = false;
    //         selfLoop = stateWithArc.first == stateWithArc.second->dstState;

    //         srcState = selfLoop ? prevTimedStateToState[stateWithArc.second->dstState] : prevTimedStateToState[stateWithArc.first];
    //         if (currTimedStateToState.find(stateWithArc.second->dstState) == currTimedStateToState.end()) {
    //             currTimedStateToState[stateWithArc.second->dstState] = dstState = id++;
    //         } else {
    //             dstState = currTimedStateToState[stateWithArc.second->dstState];
    //         }

    //         inId = stateWithArc.second->inpId;
    //         if (selfLoop) {  // self-loop
    //             outId = lmCost = 0;
    //         } else {
    //             outId = stateWithArc.second->outId;
    //             lmCost = stateWithArc.second->lmCost;
    //         }

    //         arcs.push_back({srcState, dstState, inId, outId, lmCost});
    //         if (it + 1 == rend(timeline)) {
    //             auto arcsIt = finalArcs.find(stateWithArc.second);
    //             if (arcsIt != finalArcs.end()) {
    //                 finalStates[dstState] = arcsIt->second;
    //             }
    //         }

    //         prevTimedStateToState = currTimedStateToState;
    //         currTimedStateToState.clear();
    //     }
    // }
}

void Lattice::writeAsFst(string filename, const vector<shared_ptr<Token>>& finalTokens, const vector<double>& finalTokensCost) {
    vector<Arc> arcs;
    unordered_map<uint, double> finalStates;
    constructFst(finalTokens, finalTokensCost, arcs, finalStates);

    // // connect final tokens with dummy tokens to a single final state
    // vector<unique_ptr<Token>> dummyTokens(finalStates.size());
    // for (uint i = 0; i < finalTokens.size(); i++) {
    //     dummyTokens[i] = unique_ptr<Token>(new Token());
    //     tokenToSrcState[dummyTokens[i].get()] = id--;
    //     currTokens[i] = {finalTokens[i], dummyTokens[i].get()};
    // }
    // unordered_map<const Token*, uint> commPredToCommSrcState;
    // while (currTokens.size()) {
    //     for (auto& tokensPair : currTokens) {
    //         auto& token = tokensPair.first;
    //         auto& prevToken = tokensPair.second;
    //         if (token.get()) {
    //             uint srcStateId, dstStateId;
    //             dstStateId = tokenToSrcState[prevToken];
    //             for (auto& predecessor : tokenTopology.find(token)->second.predecessors) {
    //                 if (commPredToCommSrcState.find(predecessor.get())) {
    //                     srcStateId = tokenToSrcState[token.get()] = commPredToCommSrcState[commPredToCommSrcState[predecessor.get()];
    //                 } else {
    //                     nextTokens.push_back({predecessor, token.get()});
    //                     if (token->arc == prevToken->arc) {
    //                         srcStateId = dstStateId;
    //                     } else {
    //                         srcStateId = tokenToSrcState[token.get()] = id--;
    //                     }
    //                     commPredToCommSrcState[predecessor.get()] = srcStateId;
    //                 }
    //             }
    //             assert(dstStateId);  // TODO remove that line
    //             assert(srcStateId);  // TODO remove that line
    //             arcs.push_back(Arc{srcStateId, dstStateId, token->arc->inpId, token->arc->outId, token->arc->lmCost});
    //         }
    //     }
    //     currTokens = move(nextTokens);
    // }

    // uint intialStateId = arcs.back().srcState;  // this will used to shift states so that intial state becomes zero
    // // transform states id so that srcstate is 0
    // auto transformStateId = [&](uint stateId) {
    //     if (stateId > intialStateId)
    //         return stateId - id - 1;
    //     else if (stateId < intialStateId)
    //         return stateId - id;
    //     else
    //         return (uint)0;
    // };
    // // change arcs state ids
    // for (auto& arc : arcs) {
    //     arc.srcState = transformStateId(arc.srcState);
    //     arc.dstState = transformStateId(arc.dstState);
    // }

    // // change final states
    // {
    //     unordered_map<uint, double> tmp;
    //     for (auto& keyVal : myFinalStates) {
    //         tmp[transformStateId(keyVal.first)] = keyVal.second;
    //     }
    //     myFinalStates = tmp;
    // }

    // sort arcs based on src states
    // sort(begin(arcs), end(arcs), [](const Arc& arc1, const Arc& arc2) {
    //     if (arc1.srcState != arc2.srcState) return arc1.srcState < arc2.srcState;
    //     return arc1.dstState < arc2.dstState;
    // });

    // arcs.erase(unique(begin(arcs), end(arcs), [](const Arc& a1, const Arc& a2) { return a1.srcState == a2.srcState && a1.dstState == a2.dstState &&
    //                                                                                     a1.inpId == a2.inpId && a1.outId == a2.outId; }), end(arcs));

    ofstream out;
    out.open(filename, fstream::out);
    if (!out.is_open()) {
        throw Exception("Can't open file: " + filename);
    }

    // write saved arcs and final stated
    for (uint i = 0; i < arcs.size(); ++i) {
        // write arc info
        out << arcs[i].srcState << '\t' << arcs[i].dstState << '\t';
        out << arcs[i].inpId << '\t' << arcs[i].outId;
        if (arcs[i].lmCost != 0) {
            out << '\t' << -arcs[i].lmCost;
        }
        out << endl;

        // write final state if arc destination is final state
        if (i + 1 == arcs.size() || arcs[i + 1].srcState != arcs[i].srcState) {
            auto it = finalStates.find(arcs[i].srcState + 1);
            if (it != finalStates.end()) {
                out << it->first;
                double cost = it->second;
                if (cost != 0) {
                    out << '\t' << -cost;
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
//         for (const auto& predecessor : tokenTopology[token].predecessors) {
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
//             currLeaves[curri]->predecessors.back()->nextToken = currLeaves[curri];
//             nextLeaves[nexti]->amCost = activations[inpIdsToActivationsIndx.find(nextLeaves[nexti]->arc->inpId)->second];
//             nextLeaves[nexti]->cumulativeCost = nextLeaves[nexti]->amCost + nextLeaves[nexti]->arc->lmCost + currLeaves[curri]->cumulativeCost;
//         }

//         // remove leaf if it has no predecessors
//         pruneChildLess(currLeaves[curri]);
//     }

//     // move next leaves to current leaves
//     currLeaves = move(nextLeaves);
// }
