#pragma once
#include <set>

#include "fst.hpp"
#include "helpers.hpp"

struct Token {
    uint tokId;
    const Arc* arc;
    double lmCost, amCost;
    double arcAmCost;

    Token(uint tokId, const Arc* arc, double lmCost, double amCost, double arcAmCost) : tokId(tokId), arc(arc), lmCost(lmCost), amCost(amCost), arcAmCost(arcAmCost) {}
    Token() : tokId(0), arc(0), lmCost(0), amCost(0), arcAmCost(0) {}

    void print(ostream& out) const { out << "Token id: " << tokId << " lmCost: " << lmCost << " amCost: " << amCost << " TotalCost: " << lmCost + amCost << endl; }
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
        double arcAmCost;
        Expantion(shared_ptr<Token> parentToken, double lmCost, double amCost) : parentToken(parentToken), arcAmCost(amCost) {
            this->lmCost = lmCost + (parentToken ? parentToken->lmCost : 0);
            this->amCost = amCost + (parentToken ? parentToken->amCost : 0);
        }
        Expantion() : parentToken(nullptr), lmCost(0.), amCost(0.), arcAmCost(0.) {}
    };

    struct ExpantionsComparator {
        bool operator()(const Expantion& lhs, const Expantion& rhs) const {
            return lhs.amCost + lhs.lmCost > rhs.amCost + rhs.lmCost;
        }
    };

    unordered_map<const Arc*, set<Expantion, ExpantionsComparator>> expantions;
    unordered_map<shared_ptr<Token>, Topology> tokenTopology;
    unordered_map<const Arc*, shared_ptr<Token>> arcToToken;
    uint tokenId;
    uint latticeBeam;
    double bestExpantionCost;

    void putSelfLoopFirst();
    void createFstArc(set<Arc, ArcComparator>& arcs, const shared_ptr<Token>& token, const shared_ptr<Token>& prevToken, bool isPrevTokenConnectWithMe, double accumCost, unordered_map<const Token*, uint>& tokenToSrcState, unordered_map<const Token*, uint>& tokenToDstState, uint& id);

   public:
    Lattice(uint latticeBeam, shared_ptr<Token> intialToken);
    Lattice(){};
    ~Lattice(){};

    void startNewExpantions();
    void expand(const shared_ptr<Token>& parentToken, const Arc*& arc, double lmCost, double amCost);
    void createExpandedTokens(vector<shared_ptr<Token>>& newTokens, double expantionCostBeam);
    void finishExpantions(double expantionCostBeam);

    void removeToken(shared_ptr<Token> token);
    vector<const Arc*> getBestPath(shared_ptr<Token> token);
    void latticToFst(const vector<shared_ptr<Token>>& finalTokens, const unordered_map<uint, double>& finalStates, fst::StdVectorFst* fst);
    const unordered_map<const Arc*, shared_ptr<Token>>& getArcToToken() { return arcToToken; };
};
