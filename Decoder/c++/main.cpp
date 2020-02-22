#include <ctime>
#include "fst.hpp"

int main(int argc, char const* argv[]) {
    Fst fst(BeamSearch(stoi(argv[1]), stod(argv[2])), "../DecodingGraph-large.txt", "../labels.ciphones");

    ifstream in;
    in.open("./files.txt");
    time_t t1, t2;
    time(&t1);
    string fileName;
    while (in >> fileName) {
        string shape;
        ifstream activationsFile;
        activationsFile.open("./activations/" + fileName);
        getline(activationsFile, shape);
        vector<string> dims;
        split(shape, dims);
        vector<vector<double>> activations(stoi(dims[0]), vector<double>(stoi(dims[1])));
        read2d(activationsFile, activations);
        vector<const Arc*> path = fst.decode(activations, 30.);
        for (const auto& arc : path) {
            if (!fst.isSpecialSym(arc->outLabel)) {
                cout << arc->outLabel << " ";
            }
        }
        cout << endl;
    }
    time(&t2);
    cout << "Decoding is Done in: " << t2 - t1 << " seconds \n";
    return 0;
}
