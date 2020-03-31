#include <ctime>

#include "fst.hpp"

int main(int argc, char const* argv[]) {
    time_t t1, t2;
    time(&t1);
    Fst fst(BeamSearch(5000, 0.), "../DecodingGraph_new.txt", "../labels.ciphones_new");
    time(&t2);
    cout << "Parsing is Done in: " << t2 - t1 << " seconds \n";

    ifstream in;
    in.open("./activations_test/files_test.txt");
    string fileName;
    time(&t1);
    while (in >> fileName) {
        string shape;
        ifstream activationsFile;
        activationsFile.open("./activations_test/" + fileName);
        getline(activationsFile, shape);
        vector<string> dims;
        split(shape, dims);
        vector<vector<double>> activations((uint)stoi(dims[0]), vector<double>((uint)stoi(dims[1])));
        read2d(activationsFile, activations);
        double intialRelWeight = 1.;
        for (int i = 0; i < 5; ++i) {
            vector<const Arc*> path = fst.decode(activations, intialRelWeight);
            const Arc* prefArc = NULL;
            for (const auto& arc : path) {
                // if(prefArc && arc->srcState==prefArc->srcState && arc->dstState==prefArc->dstState ){
                //     prefArc = arc;
                //     continue;
                // }
                if (!fst.isSpecialSym(arc->outLabel)) {
                    cout << arc->outLabel << " ";
                }
                prefArc = arc;
            }
            cout << endl << intialRelWeight << endl;
            intialRelWeight += 2;
        }
    }
    time(&t2);
    cout << "Decoding is Done in: " << t2 - t1 << " seconds \n";
    return 0;
}
