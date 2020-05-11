#pragma once
#include "beam_search.hpp"

struct LatticeNode {
    const Arc* arc;
    double amCost;
    double cumulativeCost;
    LatticeNode* parent;
    vector<unique_ptr<LatticeNode>> children;
    LatticeNode(const Arc* arc) : arc(arc), amCost(0), cumulativeCost(0), parent(NULL), children(vector<unique_ptr<LatticeNode>>()) {}
    LatticeNode() : arc(NULL), amCost(0), cumulativeCost(0), parent(NULL), children(vector<unique_ptr<LatticeNode>>()) {}
    LatticeNode(LatticeNode&& ln) : arc(arc), amCost(0), cumulativeCost(0), parent(NULL) { children = move(ln.children); }
};

class Lattice {
   private:
    LatticeNode root;
    LatticeNode finalNode;
    vector<LatticeNode*> currLeaves;

   public:
    Lattice(const Arc* rootArc);
    ~Lattice(){};
    void expandLattice(const vector<shared_ptr<Token>>& newTokens, const unordered_map<uint, uint>& inpIdsToActivationsIndx, const vector<double>& activations);
    void pruneChildLess(LatticeNode* node);
    void finalize();
    bool isFinalNode(const LatticeNode* node) const { return node == &finalNode; };
    const LatticeNode* getRoot() const { return &root; };
};
