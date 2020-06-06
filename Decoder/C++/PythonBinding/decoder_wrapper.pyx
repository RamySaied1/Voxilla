from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../decoder.hpp":
    cdef cppclass Decoder:
        Decoder(string ,string)
        vector[vector[string]] decode(vector[vector[double]]& activations, unsigned int maxActiveTokens, double beamWidth, double amw, unsigned int latticeBeam)
        bool isSpecialSym(string sym)

cdef class PyDecoder:
    cdef Decoder* thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, str graphFolder,str inputLabelsFile):
        self.thisptr = new Decoder(graphFolder.encode("utf-8"),inputLabelsFile.encode("utf-8"))

    def __dealloc__(self):
        del self.thisptr

    def decode(self, pyMat, maxActiveTokens, beamWidth, amw, latticeBeam=1):
        #prepare input
        cdef vector[vector[double]] cMat = vector[vector[double]](len(pyMat),vector[double](len(pyMat[0])))
        for r in range(len(pyMat)):
            for c in range(len(pyMat[0])):
                cMat[r][c] = pyMat[r][c]

        #call function
        cdef vector[vector[string]] cVec = self.thisptr.decode(cMat, maxActiveTokens, beamWidth, amw, latticeBeam)

        # prepare output
        cdef vector[vector[string]].iterator it = cVec.begin()
        pyVec = []
        while it != cVec.end():
            pyVec.append((deref(it)[0].decode("utf-8") ,deref(it)[1].decode("utf-8"), deref(it)[2].decode("utf-8")))
            inc(it)

        #return output
        return pyVec

    def is_special_sym(self,str sym):
        return self.thisptr.isSpecialSym(sym.encode("utf-8"))