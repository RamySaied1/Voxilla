#pragma once
#include "helpers.hpp"
struct Arc;

struct Token {
    uint tokId;
    const Arc* arc;
    double lmCost, amCost, hmmCost;

    Token(uint tokId, const Arc* arc, double lmCost, double amCost, double hmmCost) : tokId(tokId), arc(arc), lmCost(lmCost), amCost(amCost), hmmCost(hmmCost) {}
    Token() : tokId(0), arc(0), lmCost(0), amCost(0) {}

    void print(ostream& out) const { out << "Token id: " << tokId << " lmCost: " << lmCost << " amCost: " << amCost << " hmmCost: " << hmmCost << endl; }
    // bool operator==(const Token& other) const { return tokId == other.tokId && node == other.node; }
};

class BeamSearch {
   public:
    BeamSearch(uint beamWidth, double pathAcceptingThreshold = 0.);
    ~BeamSearch();

    const vector<shared_ptr<Token>>& getExpandedTokens() const;
    const vector<shared_ptr<Token>>& getActiveTokens() const;
    void setRootToken(const Arc* arc, double lmCost, double modelScre, double hmmCost);
    void setActiveTokens(const vector<shared_ptr<Token>>& tokens);
    void moveExpandedToActive();
    void beamPrune(std::function<double(double, double, double)> combineAmLmHmmCosts);
    void keepOnlyBestExpandedTokens(std::function<double(double, double, double)> combineAmLmHmmCosts);
    void doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, std::function<double(double, double, double)> combineAmLmHmmCosts, bool useSelfLoops);
    vector<const Arc*> getBestPath(const vector<vector<const Arc*>>& graph, function<double(double, double, double)> combineAmLmHmmCosts, Token& bestToken);

   private:
    struct Expantion {
        shared_ptr<Token> parentToken;
        double lmCost, amCost, hmmCost, expantionCost;
        Expantion(shared_ptr<Token> parentToken, double lmCost, double amCost, double hmmCost, double expantionCost) : parentToken(parentToken), lmCost(lmCost), amCost(amCost), hmmCost(hmmCost), expantionCost(expantionCost) {}
        Expantion() : parentToken(NULL), lmCost(0.), amCost(0.), expantionCost(0.) {}
    };

    uint beamWidth;
    double pathAcceptingThreshold;
    vector<shared_ptr<Token>> activeTokens, expandedTokens;
    unordered_map<shared_ptr<Token>, shared_ptr<Token>> predeccessor;

    vector<double> getNormalizeTokensLogProba(const vector<shared_ptr<Token>>& tokens, function<double(double, double, double)> combineAmLmHmmCosts);
    void createExpandedTokens(const unordered_map<const Arc*, Expantion>& expantions);
    void expandNewToken(const Arc* arc, double lmCost, double amCost, double hmmCost) {
        static uint tokenId = 1;
        expandedTokens.push_back(shared_ptr<Token>(new Token(tokenId, arc, lmCost, amCost, hmmCost)));
        ++tokenId;
    }
};
