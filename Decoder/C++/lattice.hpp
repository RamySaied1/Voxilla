#pragma once
#include "helpers.hpp"
#include "fst.hpp"


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
        uint successorsCount = 0;
    };

    struct Expantion {
        shared_ptr<Token> parentToken;
        double lmCost, amCost;
        Expantion(shared_ptr<Token> parentToken, double lmCost, double amCost) : parentToken(parentToken), lmCost(lmCost), amCost(amCost) {}
        Expantion() : parentToken(NULL), lmCost(0.), amCost(0.) {}
    };

    shared_ptr<Token> root;
    unordered_map<const Arc*, vector<Expantion>> expantions;
    unordered_map<shared_ptr<Token>, Topology> tokenToplogy;
    vector<shared_ptr<Token>> lastlyExpandedTokens;
    uint tokenId = 1;

   public:
    Lattice();
    ~Lattice(){};
    void startNewExpantions();
    void expand(const shared_ptr<Token>& parent, const Arc*& arc, double lmCost, double amCost);
    void finishExpantions(vector<shared_ptr<Token>> & newToken);
    vector<shared_ptr<Token>> getBestPath(shared_ptr<Token> token);
    void removeToken(shared_ptr<Token> token);

    // void pruneChildLess(LatticeNode* node);
    // void finalize();
    // bool isFinalNode(const LatticeNode* node) const { return node == &finalNode; };
    // const LatticeNode* getRoot() const { return &root; };
};
