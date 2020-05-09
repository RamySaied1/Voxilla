#include <ctime>
#include "decoder.hpp"

int main(int argc, char const* argv[]) {
    string graphFolder = argv[1];
    string activationsFolder = argv[2];
    string filesFile = argv[3];
    uint maxActiveTokens = (uint)stoi(argv[4]);
    double beamWidth = stod(argv[5]);

    time_t t1, t2;
    time(&t1);
    Decoder decoder(graphFolder, graphFolder + "labels.ciphones");
    time(&t2);
    cout << "Parsing is Done in: " << t2 - t1 << " seconds \n";

    ifstream in;
    in.open(activationsFolder + filesFile);
    string fileName;
    time(&t1);
    double amwStart = stod(argv[6]);
    double amwEnd = stod(argv[7]);
    double amwStep = stod(argv[8]);
    vector<vector<double>> activations;
    for (double amw = amwStart; amw <= amwEnd; amw += amwStep) {
        long long totalFramesNum = 0;
        uint filesCount = 0;
        time(&t1);
        while (in >> fileName) {
            ++filesCount;
            string shape;
            ifstream activationsFile;
            activationsFile.open(activationsFolder + fileName);
            getline(activationsFile, shape);
            vector<string> dims;
            split(shape, dims);
            activations = vector<vector<double>>((uint)stoi(dims[0]), vector<double>((uint)stoi(dims[1])));
            totalFramesNum += activations.size();
            read2d(activationsFile, activations);
            
            decoder.decode(activations, maxActiveTokens, beamWidth, 1 / amw);
            for(auto & path: decoder.getBestNPath(10)){
                for (int i = 0; i < path.size(); ++i) {
                    if (i && path[i].back() == path[i - 1].back()) {
                        continue;
                    }
                    if (!decoder.isSpecialSym(path[i][1])) {
                        cout << path[i][1] << " ";
                    }
                }
                cout << endl;
            }
        }
        cout << "amw: " << amw << endl;
        time(&t2);
        cout << "Decoding " << filesCount << " files with " << totalFramesNum << " frames is Done in: " << t2 - t1 << " seconds "
             << " Avg: " << (t2 - t1) * 1. / totalFramesNum << " frame/sec" << endl;
        in.clear();
        in.seekg(0, ios::beg);
    }

    return 0;
}
