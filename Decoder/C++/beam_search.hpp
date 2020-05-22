#pragma once
#include "lattice.hpp"

struct Arc;

class BeamSearch {
   public:
    BeamSearch(uint maxActiveTokens = 500, double beamWidth = 0.);
    ~BeamSearch();

    // core algoirthm functions
    void intiate(const Arc* arc, double lmCost, double modelScre, uint maxActiveTokens, double beamWidth, uint latticeBeam, const Fst* fst);
    void doForward(const unordered_map<uint, uint>& inpIdsToIndx, const vector<double>& activations, bool useSelfLoops);
    void moveExpandedToActive();
    void beamPrune();
    void expandEpsStates();
    void applyFinalState(const unordered_map<uint, double>& finalStates);
    vector<const Arc*> getBestPath(Token& bestToken);
    
    // lattice functions
    void startNewExpantions() { lattice.startNewExpantions(); }
    void createExpandedTokens() { lattice.createExpandedTokens(expandedTokens, beamWidth); }
    void finishExpantions() { lattice.finishExpantions(beamWidth); }
    
    
    // basic class functions
    void setActiveTokens(const vector<shared_ptr<Token>>& tokens);
    Lattice* getLattice() { return &lattice; }
    const vector<shared_ptr<Token>>& getExpandedTokens() const;
    const vector<shared_ptr<Token>>& getActiveTokens() const;

   private:
    uint maxActiveTokens;
    double beamWidth;
    vector<shared_ptr<Token>> activeTokens, expandedTokens;
    Lattice lattice;
    const Fst* fst;
};
