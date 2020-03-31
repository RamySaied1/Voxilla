#include <ctime>

#include "fst.hpp"

int main(int argc, char const* argv[]) {
    time_t t1, t2;
    time(&t1);
    Fst fst(BeamSearch(1000, 0.), "../DecodingGraph_new.txt", "../labels.ciphones_new");
    time(&t2);
    cout << "Parsing is Done in: " << t2 - t1 << " seconds \n";

    ifstream in;
    in.open("./activations_test/files.txt");
    time(&t1);
    string fileName;
    int enought = 0;
    while (in >> fileName) {
        string shape;
        ifstream activationsFile;
        activationsFile.open("./activations_test/" + fileName);
        getline(activationsFile, shape);
        vector<string> dims;
        split(shape, dims);
        vector<vector<double>> activations((uint)stoi(dims[0]), vector<double>((uint)stoi(dims[1])));
        read2d(activationsFile, activations);
        double intialRelWeight = .25;
        for (int i = 0; i < 40; ++i) {
            vector<const Arc*> path = fst.decode(activations, intialRelWeight);
            for (const auto& arc : path) {
                if (!fst.isSpecialSym(arc->outLabel)) {
                    cout << arc->outLabel << " ";
                }
            }
            cout << endl
                 << intialRelWeight << endl;
            intialRelWeight += .25;
        }
        enought += 1;
        if (enought == 2) {
            break;
        }
    }
    time(&t2);
    cout << "Decoding is Done in: " << t2 - t1 << " seconds \n";
    return 0;
}
