#include "helpers.hpp"

void split(string str, vector<string>& fields) {
    istringstream iss(str);
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(fields));
}

void strtolower(string& s) { transform(begin(s), end(s), begin(s), ::tolower); }

void writeFst(fst::StdVectorFst& fst, string filename) {
    ofstream out;
    out.open(filename, ifstream::out);
    if (!out.is_open()) {
        throw Exception("Can't open/create: " + filename);
    }

    fst::StdArc::StateId stateId;
    for (fst::StateIterator<fst::StdFst> siter(fst); !siter.Done(); siter.Next()) {
        stateId = siter.Value();
        fst::StdArc::Weight finalWeight = fst.Final(stateId);
        if (finalWeight != fst::StdArc::Weight::Zero()) {
            out << stateId;
            if (finalWeight.Value()) out << "\t" << finalWeight.Value();
            out << endl;
        }
        for (fst::ArcIterator<fst::StdFst> aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
            const fst::StdArc& arc = aiter.Value();
            out << stateId << "\t" << arc.nextstate << "\t" << arc.ilabel << "\t" << arc.olabel;
            if (arc.weight.Value()) out << "\t" << arc.weight.Value();
            out << endl;
        }
    }
}