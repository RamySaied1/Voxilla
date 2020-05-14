#pragma once
#include <set>

#include "fst.hpp"
#include "helpers.hpp"

struct Token {
    uint tokId;
    const Arc* arc;
    double lmCost, amCost;

    Token(uint tokId, const Arc* arc, double lmCost, double amCost) : tokId(tokId), arc(arc), lmCost(lmCost), amCost(amCost) {}
    Token() : tokId(0), arc(0), lmCost(0), amCost(0) {}

    void print(ostream& out) const { out << "Token id: " << tokId << " lmCost: " << lmCost << " amCost: " << amCost << endl; }
    // bool operator==(const Token& other) const { return tokId == other.tokId && node == other.node; }
};

class Lattice {
   private:
    struct Topology {
        vector<shared_ptr<Token>> predecessors;
        uint successorsCount;
        Topology() : predecessors(vector<shared_ptr<Token>>()), successorsCount(0) {}
    };

    struct Expantion {
        shared_ptr<Token> parentToken;
        double lmCost, amCost;
        Expantion(shared_ptr<Token> parentToken, double lmCost, double amCost) : parentToken(parentToken) {
            this->lmCost = lmCost + (parentToken ? parentToken->lmCost : 0);
            this->amCost = amCost + (parentToken ? parentToken->amCost : 0);
        }
        Expantion() : parentToken(NULL), lmCost(0.), amCost(0.) {}
    };

    struct ExpantionsComparator {
        bool operator()(const Expantion& lhs, const Expantion& rhs) const {
            return lhs.amCost + lhs.lmCost > rhs.amCost + rhs.lmCost;
        }
    };

    unordered_map<const Arc*, set<Expantion, ExpantionsComparator>> expantions;
    unordered_map<shared_ptr<Token>, Topology> tokenToplogy;
    unordered_map<const Arc*, shared_ptr<Token>> arcToToken;
    uint tokenId;
    uint latticeBeam;

   public:
    Lattice(uint latticeBeam);
    ~Lattice(){};

    void startNewExpantions();
    void expand(const shared_ptr<Token>& parent, const Arc*& arc, double lmCost, double amCost);
    void createExpandedTokens(vector<shared_ptr<Token>>& newTokens);
    void finishExpantions();

    vector<shared_ptr<Token>> getBestPath(shared_ptr<Token> token);
    void removeToken(shared_ptr<Token> token);

    // void pruneChildLess(LatticeNode* node);
    // void finalize();
    // bool isFinalNode(const LatticeNode* node) const { return node == &finalNode; };
    // const LatticeNode* getRoot() const { return &root; };
};
