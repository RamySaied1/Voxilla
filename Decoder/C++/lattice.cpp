#include "lattice.hpp"

Lattice::Lattice(uint latticeBeam) : latticeBeam(latticeBeam), tokenId(1) {
}

void Lattice::startNewExpantions() {
    expantions.clear();
    arcToToken.clear();
}

void Lattice::expand(const shared_ptr<Token>& parent, const Arc*& arc, double lmCost, double amCost) {
    expantions[arc].insert(Expantion(parent, lmCost, amCost));
};

void Lattice::createExpandedTokens(vector<shared_ptr<Token>>& newTokens) {
    for (auto& keyVal : expantions) {
        const auto& arc = keyVal.first;
        auto& expantionsSet = keyVal.second;
        double lmCost = expantionsSet.begin()->lmCost;
        double amCost = expantionsSet.begin()->amCost;

        if (arcToToken.find(arc) != arcToToken.end()) {  // this token created before in same time step
            arcToToken[arc]->lmCost = lmCost;
            arcToToken[arc]->amCost = amCost;
        } else {
            shared_ptr<Token> newToken = shared_ptr<Token>(new Token(tokenId++, arc, lmCost, amCost));
            newTokens.push_back(newToken);
            arcToToken[arc] = newToken;
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

void Lattice::getAllPathes(shared_ptr<Token> token, vector<vector<const Arc*>> & pathes) {
    static vector<const Arc*> arcSeq;
    if (!token) {
        pathes.push_back(arcSeq);
        reverse(begin(pathes.back()), end(pathes.back()));
    } else {
        arcSeq.push_back(token->arc);
        for (const auto& predecessor : tokenToplogy[token].predecessors) {
            getAllPathes(predecessor, pathes);
        }
        arcSeq.pop_back();
    }
}

void Lattice::pruneChildLess(shared_ptr<Token> token) {
    if (token.get() && tokenToplogy[token].successorsCount == 0) {
        vector<shared_ptr<Token>> predecessors = tokenToplogy[token].predecessors;
        tokenToplogy.erase(token);
        for (auto& predecessor : predecessors) {
            tokenToplogy[predecessor].successorsCount -= 1;
            if (tokenToplogy[predecessor].successorsCount == 0) {
                pruneChildLess(predecessor);
            }
        }
    }
}

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

// void Lattice::pruneChildLess(LatticeNode* node) {
//     if (node->predecessors.size() == 0) {
//         LatticeNode* parent = node->parent;
//         auto endit = remove_if(begin(parent->predecessors), end(parent->predecessors), [&](const auto& nodePtr) {
//             return nodePtr.get() == node;
//         });
//         parent->predecessors.erase(endit, end(parent->predecessors));
//         pruneChildLess(parent);
//     }
// }