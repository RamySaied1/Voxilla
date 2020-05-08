#pragma once
#include "helpers.hpp"
struct Arc;

struct Token {
    uint tokId;
    const Arc* arc;
    double lmCost, amCost;

    Token(uint tokId, const Arc* arc, double lmCost, double amCost) : tokId(tokId), arc(arc), lmCost(lmCost), amCost(amCost) {}
    Token() : tokId(0), arc(0), lmCost(0), amCost(0) {}

    void print(ostream& out) const { out << "Token id: " << tokId << " lmCost: " << lmCost << " amCost: " << amCost << endl; }
    // bool operator==(const Token& other) const { return tokId == other.tokId && node == other.node; }
};

class BeamSearch {
   public:
    BeamSearch(uint maxActiveTokens = 500, double beamWidth = 0.);
    ~BeamSearch();

    const vector<shared_ptr<Token>>& getExpandedTokens() const;
    const vector<shared_ptr<Token>>& getActiveTokens() const;
    void intiate(const Arc* arc, double lmCost, double modelScre, uint maxActiveTokens, double beamWidth);
    void setActiveTokens(const vector<shared_ptr<Token>>& tokens);
    void moveExpandedToActive();
    void beamPrune();
    void keepOnlyBestExpandedTokens();
    void doForward(const vector<vector<const Arc*>>& graph, const unordered_map<uint, uint>& inpIdsToIndx, const vector<double>& activations, bool useSelfLoops);
    vector<const Arc*> getBestPath(Token& bestToken);
    vector<vector<const Arc*>> getBestNPath(uint N);

   private:
    struct Expantion {
        shared_ptr<Token> parentToken;
        double lmCost, amCost, expantionCost;
        Expantion(shared_ptr<Token> parentToken, double lmCost, double amCost, double expantionCost) : parentToken(parentToken), lmCost(lmCost), amCost(amCost), expantionCost(expantionCost) {}
        Expantion() : parentToken(NULL), lmCost(0.), amCost(0.), expantionCost(0.) {}
    };

    uint maxActiveTokens;
    double beamWidth;
    vector<shared_ptr<Token>> activeTokens, expandedTokens;
    unordered_map<shared_ptr<Token>, shared_ptr<Token>> predeccessor;

    vector<double> getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens);
    void createExpandedTokens(const unordered_map<const Arc*, Expantion>& expantions);
    void expandNewToken(const Arc* arc, double lmCost, double amCost) {
        static uint tokenId = 1;
        expandedTokens.push_back(shared_ptr<Token>(new Token(tokenId, arc, lmCost, amCost)));
        ++tokenId;
    }
};
