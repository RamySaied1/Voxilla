#pragma once
#include "helpers.hpp"
struct Arc;

struct Token {
    uint tokId;
    const Arc* arc;
    double lmScore, modelScore;

    Token(uint tokId, const Arc* arc, double lmScore, double modelScore) : tokId(tokId), arc(arc), lmScore(lmScore), modelScore(modelScore) {}
    Token() : tokId(0), arc(0), lmScore(0), modelScore(0) {}

    void print(ostream& out) const { out << "Token id: " << tokId << " lmScore: " << lmScore << " modelScore: " << modelScore << " Joint score:" << modelScore + lmScore << endl; }
    // bool operator==(const Token& other) const { return tokId == other.tokId && node == other.node; }
};

class BeamSearch {
   public:
    BeamSearch(uint beamWidth);
    ~BeamSearch();

    const vector<shared_ptr<Token>>& getExpandedTokens() const;
    const vector<shared_ptr<Token>>& getActiveTokens() const;
    void setRootToken(const Arc* arc, double lmScore, double modelScre);
    void setActiveTokens(const vector<shared_ptr<Token>>& tokens);
    void moveExpandedToActive();
    void beamPrune();
    void applyFinalState(const unordered_map<uint, double>& finalStates);
    vector<const Arc*> getBestPath(const vector<vector<const Arc*>>& graph, Token& bestToken);
    void keepOnlyBestExpandedTokens();
    void doForward(const vector<vector<const Arc*>>& graph, const unordered_map<string, uint>& inpLabelsToIndx, const vector<double>& activations, bool useSelfLoops);

   private:
    struct Expantion {
        shared_ptr<Token> parentToken;
        double lmScore, modelScore;
        Expantion(shared_ptr<Token> parentToken, double lmScore, double modelScore) : parentToken(parentToken), lmScore(lmScore), modelScore(modelScore) {}
        Expantion() : parentToken(NULL), lmScore(0.), modelScore(0.) {}
    };

    int beamWidth;
    unordered_map<shared_ptr<Token>, shared_ptr<Token>> predeccessor;
    vector<shared_ptr<Token>> activeTokens, expandedTokens;

    vector<double> getNormalizeTokensProba(const vector<shared_ptr<Token>>& tokens);
    void createExpandedTokens(const unordered_map<const Arc*, Expantion>& expantions);
    void expandNewToken(const Arc* arc, double lmScore, double modelScore) {
        static uint tokenId = 1;
        expandedTokens.push_back(shared_ptr<Token>(new Token(tokenId, arc, lmScore, modelScore)));
        ++tokenId;
    }
};
