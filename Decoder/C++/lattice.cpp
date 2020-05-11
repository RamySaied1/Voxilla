#include "lattice.hpp"

#include "fst.hpp"

Lattice::Lattice(const Arc* rootArc) : root(LatticeNode(rootArc)) {
    this->currLeaves = vector<LatticeNode*>();
    this->currLeaves.push_back(&root);
}

void Lattice::expandLattice(const vector<shared_ptr<Token>>& newTokens, const unordered_map<uint, uint>& inpIdsToActivationsIndx, const vector<double>& activations) {
    vector<const Arc*> newArcs = vector<const Arc*>(newTokens.size());
    transform(begin(newTokens), end(newTokens), begin(newArcs), [](const auto& token) { return token->arc; });

    // sort current leaves on dst state
    sort(begin(currLeaves), end(currLeaves), [](const auto& l1, const auto& l2) {
        return l1->arc->dstState < l2->arc->dstState;
    });

    // create next leaves
    vector<LatticeNode*> nextLeaves(newArcs.size());
    for (size_t i = 0; i < nextLeaves.size(); i++) {
        nextLeaves[i] = new LatticeNode(newArcs[i]);
    }

    // sort next leaves on src state
    sort(begin(nextLeaves), end(nextLeaves), [](const auto& l1, const auto& l2) {
        return l1->arc->srcState < l2->arc->srcState;
    });

    // connect currLeaves to next leaves
    for (size_t curri = 0, nexti = 0; curri < nextLeaves.size(); curri++) {
        for (; nexti < nextLeaves.size() && nextLeaves[nexti]->arc->srcState == currLeaves[curri]->arc->dstState; ++nexti) {
            currLeaves[curri]->children.push_back(unique_ptr<LatticeNode>(nextLeaves[nexti]));
            currLeaves[curri]->children.back()->parent = currLeaves[curri];
            nextLeaves[nexti]->amCost = activations[inpIdsToActivationsIndx.find(nextLeaves[nexti]->arc->inpId)->second];
            nextLeaves[nexti]->cumulativeCost = nextLeaves[nexti]->amCost + nextLeaves[nexti]->arc->lmCost + currLeaves[curri]->cumulativeCost;
        }

        // remove leaf if it has no children
        pruneChildLess(currLeaves[curri]);
    }

    // move next leaves to current leaves
    currLeaves = move(nextLeaves);
}

void Lattice::pruneChildLess(LatticeNode* node) {
    if (node->children.size() == 0) {
        LatticeNode* parent = node->parent;
        auto endit = remove_if(begin(parent->children), end(parent->children), [&](const auto& nodePtr) {
            return nodePtr.get() == node;
        });
        parent->children.erase(endit, end(parent->children));
        pruneChildLess(parent);
    }
}