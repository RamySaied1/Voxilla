#include <ctime>
#include "fst.hpp"

int main(int argc, char const* argv[]) {
    time_t t1, t2;

    time(&t1);
    Fst fst(BeamSearch(500), "../DecodingGraph.txt", "labels.ciphones");
    time(&t2);
    // cout << "Parsing Done in: " << t2 - t1 << " seconds \n";

    ifstream in;
    in.open("./files.txt");

    string fileName;
    while (in >> fileName) {
        string shape;
        ifstream activationsFile;
        activationsFile.open(fileName);
        getline(activationsFile, shape);
        vector<string> dims;
        split(shape, dims);
        vector<vector<double>> activations(stoi(dims[0]), vector<double>(stoi(dims[1])));
        read2d(activationsFile, activations);

        // cout << "Reading Activations Done\n";
        // time(&t1);
        vector<const Arc*> path = fst.decode(activations, 30.);
        // time(&t2);
        // cout << "Decoding is Done in: " << t2 - t1 << " seconds \n";

        for (const auto& arc : path) {
            if (!fst.isSpecialSym(arc->outLabel)) {
                cout << arc->outLabel << " ";
            }
        }
        cout << endl;
    }

    return 0;
}
