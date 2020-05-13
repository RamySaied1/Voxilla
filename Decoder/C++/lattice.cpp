#include "lattice.hpp"

Lattice::Lattice() {
}

void Lattice::startNewExpantions() {
    expantions.clear();
    lastlyExpandedTokens.clear();
}

void Lattice::expand(const shared_ptr<Token>& parent, const Arc*& arc, double lmCost, double amCost) {
    expantions[arc].push_back(Expantion(parent, lmCost, amCost));
};

void Lattice::finishExpantions(vector<shared_ptr<Token>>& newTokens) {
    for (auto& keyVal : expantions) {
        uint topK = 5;
        const auto& arc = keyVal.first;
        auto& expantionsVector = keyVal.second;
        shared_ptr<Token> newToken = shared_ptr<Token>(new Token(tokenId++, arc, 0, 0));
        newTokens.push_back(newToken);

        sort(begin(expantionsVector), end(expantionsVector), [](const Expantion& expan1, const Expantion& expan2) {
            return expan1.amCost + expan1.lmCost < expan2.amCost + expan2.lmCost;
        });

        for (uint i = 0; i < min(topK, (uint)expantionsVector.size()); ++i) {
            double amCost = expantionsVector[i].amCost + (expantionsVector[i].parentToken.get() ? expantionsVector[i].parentToken->amCost : 0);
            double lmCost = expantionsVector[i].lmCost + (expantionsVector[i].parentToken.get() ? expantionsVector[i].parentToken->lmCost : 0);
            if (!i) {
                newToken->amCost = amCost;
                newToken->lmCost = lmCost;
            }
            tokenToplogy[newToken].predecessors.push_back(expantionsVector[i].parentToken);
            if(expantionsVector[i].parentToken.get())
                tokenToplogy[expantionsVector[i].parentToken].successorsCount += 1;
        }
    }
    expantions.clear();
}

vector<shared_ptr<Token>> Lattice::getBestPath(shared_ptr<Token> token) {
    vector<shared_ptr<Token>> tokenSeq;
    shared_ptr<Token> currToken = token;
    while (currToken) {
        tokenSeq.push_back(token);
        currToken = tokenToplogy[currToken].predecessors.front();
    }

    reverse(begin(tokenSeq), end(tokenSeq));
    return tokenSeq;
}

void Lattice::removeToken(shared_ptr<Token> token) {
    assert(tokenToplogy[token].successorsCount == 0);
    auto& predecessors = tokenToplogy[token].predecessors;
    tokenToplogy.erase(token);
    for (auto& child : predecessors) {
        tokenToplogy[child].successorsCount -= 1;
        if (tokenToplogy[child].successorsCount == 0) {
            removeToken(child);
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