#include "fst.hpp"
#include <exception>

Fst::Fst(BeamSearch decoder, string fstFileName, string labelsFileName, SpecialSymbols espSyms) : espSyms(espSyms), decoder(decoder) {
    inpLabelToIndx = unordered_map<string, uint>();
    parseInputLabels(labelsFileName);

    graph = vector<vector<const Arc*>>();
    finalStates = unordered_map<uint, double>();
    parseFst(fstFileName);
    preprocessFst();
}

void Fst::parseInputLabels(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);

    try {
        string line;
        int i = 0;
        while (in >> line) {
            if (isSpecialSym(line)) {
                throw Exception("special symbol exists in labels file which is not expected");
            } else {
                inpLabelToIndx[line] = i++;
            }
        }
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        inpLabelToIndx.clear();
    }
    in.close();
}

void Fst::parseFst(const string& filename) {
    ifstream in;
    in.open(filename, ifstream::in);
    try {
        string line;
        while (getline(in, line)) {
            vector<string> fields;
            split(line, fields);

            if (fields.size() <= 2) {
                processFinalState(fields);
            } else {
                processArc(fields);
            }
        }
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        graph.clear();
    }
    in.close();
}

void Fst::processFinalState(const vector<string>& fields) {
    int finalState = stoi(fields[0]);
    double cost = (fields.size() == 2) ? -stod(fields.back()) : 0.;
    finalStates[finalState] = cost;
    if (finalState == graph.size()) {
        graph.push_back(vector<const Arc*>());
    }
}

void Fst::processArc(const vector<string>& fields) {
    int srcState = stoi(fields[0]);
    const Arc* arc = new Arc{srcState, stoi(fields[1]), fields[2], fields[3], (fields.size() == 5) ? -stod(fields.back()) : 0.};

    if (srcState < graph.size()) {
        graph[srcState].push_back(arc);
    } else {
        graph.push_back(vector<const Arc*>({arc}));
    }
}

void Fst::preprocessFst() {
    graph.reserve(graph.size());
    for (auto& Arcs : graph) {
        moveRelevantFisrt<const Arc*>(Arcs, [&](auto arc) {  // put speical symbols at first
            return arc->inpLabel == espSyms.epsSymbol;
        });
        Arcs.reserve(Arcs.size());  // reclaim useless space
    }
}

void Fst::expandEpsStates() {
    const vector<shared_ptr<Token>>& expandedTokens = decoder.getExpandedTokens();
    int i = 0;
    while (i < expandedTokens.size()) {
        vector<shared_ptr<Token>> epsTokens;
        for (; i < expandedTokens.size(); ++i) {
            const auto& token = expandedTokens[i];
            if (hasEpsArc(token->arc->dstState)) {
                epsTokens.push_back(token);
            }
        }
        decoder.setActiveTokens(epsTokens);
        decoder.doForward(graph, {{espSyms.epsSymbol, 0}}, vector<double>(1, 0.), false);
    }

    decoder.keepOnlyBestExpandedTokens();
}

vector<const Arc*> Fst::decode(vector<vector<double>>& activations, double lmWeight) {
    preprocessActivations(activations, lmWeight);  // normalize activations and apply lm relative weight
    unique_ptr<Arc> intialArc(new Arc{-1, 0, espSyms.epsSymbol, espSyms.epsSymbol, 0.});
    decoder.setRootToken(intialArc.get(), 0., 0.);

    for (const auto& row : activations) {
        decoder.doForward(graph, inpLabelToIndx, row, true);
        decoder.beamPrune();
        expandEpsStates();
        decoder.moveExpandedToActive();
    }

    decoder.applyFinalState(finalStates);
    Token finalToken = Token();
    auto path = decoder.getBestPath(graph, finalToken);
    finalToken.print(cout);
    return move(path);
}

void Fst::preprocessActivations(vector<vector<double>>& activations, double relativeWeight) {
    for (auto& exLogProbas : activations) {
        double maxVal = *max_element(begin(exLogProbas), end(exLogProbas));
        for (auto& elem : exLogProbas) {
            elem = (elem - maxVal) / relativeWeight;
        }
    }
}

pair<int, double> Fst::skipStartNodes() {
    double cost = 0;
    int rootNode = 0;
    while (graph[rootNode].size() == 1 && graph[rootNode].front()->inpLabel == espSyms.startSymbol) {
        cost += graph[rootNode].front()->cost;
        rootNode = graph[rootNode].front()->dstState;
    }
    return {rootNode, cost};
}

Fst::~Fst() {
    for (auto& arcs : graph) {
        for (auto& arc : arcs) {
            delete arc;
        }
    }
}