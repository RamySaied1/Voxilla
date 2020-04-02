#include <ctime>

#include "decoder.hpp"

int main(int argc, char const* argv[]) {
    time_t t1, t2;
    time(&t1);
    Decoder decoder("../graphs/New/HCLG.txt", "../graphs/New/labels.ciphones", (uint)stoi(argv[1]), stod(argv[2]));
    time(&t2);
    // cout << "Parsing is Done in: " << t2 - t1 << " seconds \n";

    string activationsFolder = "./DNN1_activations_test/";
    ifstream in;
    in.open(activationsFolder + "files.txt");
    string fileName;
    time(&t1);
    int amwMax = stoi(argv[3]);
    int lmwMax = stoi(argv[4]);
    int hmmwMax = stoi(argv[5]);
    for (size_t lmw = 1; lmw <= lmwMax; lmw=ceil(lmw*1.75)) {
        for (size_t hmmw = 1; hmmw <= hmmwMax; hmmw=ceil(hmmw*1.75)) {
            for (size_t amw = 1; amw <= amwMax; amw=ceil(amw*1.75)) {
                while (in >> fileName) {
                    string shape;
                    ifstream activationsFile;
                    activationsFile.open(activationsFolder + fileName);
                    getline(activationsFile, shape);
                    vector<string> dims;
                    split(shape, dims);
                    vector<vector<double>> activations((uint)stoi(dims[0]), vector<double>((uint)stoi(dims[1])));
                    read2d(activationsFile, activations);

                    vector<vector<string>> path = decoder.decode(activations, amw, lmw, hmmw);
                    for (int i = 0; i < path.size(); ++i) {
                        if (i && path[i].back() == path[i - 1].back()) {
                            continue;
                        }
                        if (path[i][1] != "<eps>") {
                            cout << path[i][1] << " ";
                        }
                    }
                    cout <<"."<< endl;
                }
                cout<<"lmw: " << lmw << " amw: " << amw << " hmmw: " << hmmw << endl;
                in.clear();
                in.seekg(0, ios::beg);
            }
        }
    }

    return 0;
}
