#pragma once
#include "lattice.hpp"

struct Arc;

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
    vector<vector<const Arc*>> getBestNPath(uint n);
    void applyFinalState(const unordered_map<uint, double>& finalStates);
    void startNewExpantions() { lattice.startNewExpantions(); }
    void createExpandedTokens() { lattice.createExpandedTokens(expandedTokens); }
    void finishExpantions() { lattice.finishExpantions(); }

   private:
    uint maxActiveTokens;
    double beamWidth;
    vector<shared_ptr<Token>> activeTokens, expandedTokens;
    Lattice lattice;
};
