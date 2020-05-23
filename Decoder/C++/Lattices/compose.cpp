#include <fst/fstlib.h>

#include <iostream>
#include <memory>

using namespace std;
using namespace fst;

int main(int argc, char const* argv[]) {
    string latFile, gFile;
    cin >> gFile;

    VectorFst<LogArc>* gFst = VectorFst<LogArc>::Read(gFile);
    string newFileName;
    VectorFst<LogArc>* latFst;
    while (cin >> latFile) {
        latFst = VectorFst<LogArc>::Read(latFile);
        VectorFst<LogArc> result;
        Compose(*latFst, *gFst, &result);
        newFileName = "composed_" + latFile;
        result.Write(newFileName);
        delete latFst;
        cout << newFileName << "\t DONE\n";
    }

    return 0;
}
