#include <fst/fstlib.h>

#include <iostream>
#include <memory>

using namespace std;
using namespace fst;

int main(int argc, char const* argv[]) {
    string latFile, gFile;
    cin >> gFile;

    StdVectorFst* gFst = StdVectorFst::Read(gFile);
    string newFileName;
    StdVectorFst* latFst;
    while (cin >> latFile) {
        latFst = StdVectorFst::Read(latFile);
        StdVectorFst result;
        Compose(*latFst, *gFst, &result);
        newFileName = "composed_" + latFile;
        result.Write(newFileName);
        delete latFst;
        cout << newFileName << "\t DONE\n";
    }

    return 0;
}
