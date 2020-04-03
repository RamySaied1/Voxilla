#include <ctime>
#include "decoder.hpp"

int main(int argc, char const* argv[]) {
    time_t t1, t2;
    time(&t1);
    Decoder decoder("../graphs/New/HCLG.txt", "../graphs/New/labels.ciphones", (uint)stoi(argv[1]), stod(argv[2]));
    time(&t2);
    // cout << "Parsing is Done in: " << t2 - t1 << " seconds \n";

    string activationsFolder = "./BLSTM_activations_test/";
    ifstream in;
    in.open(activationsFolder + "files.txt");
    string fileName;
    time(&t1);
    int amwMax = stoi(argv[3]);
    for (float amw = 1; amw <= amwMax; amw += .5) {
        while (in >> fileName) {
            string shape;
            ifstream activationsFile;
            activationsFile.open(activationsFolder + fileName);
            getline(activationsFile, shape);
            vector<string> dims;
            split(shape, dims);
            vector<vector<double>> activations((uint)stoi(dims[0]), vector<double>((uint)stoi(dims[1])));
            read2d(activationsFile, activations);

            vector<vector<string>> path = decoder.decode(activations, 1 / amw);
            for (int i = 0; i < path.size(); ++i) {
                if (i && path[i].back() == path[i - 1].back()) {
                    continue;
                }
                if (path[i][1] != "<eps>") {
                    cout << path[i][1] << " ";
                }
            }
            cout << "." << endl;
        }
        cout <<" amw: " << endl;
        in.clear();
        in.seekg(0, ios::beg);
    }

    return 0;
}
