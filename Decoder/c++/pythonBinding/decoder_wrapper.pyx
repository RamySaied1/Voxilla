from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "../decoder.cpp":
    cdef cppclass Decoder:
        Decoder(string,string,int,double)
        vector[vector[string]] decode(vector[vector[double]]& activations, double lmWeight)

cdef extern from "../beam_search.cpp":
    cdef cppclass BeamSearch:
        BeamSearch(unsigned int , double)

cdef extern from "../fst.cpp":
    cdef cppclass Fst:
        Fst(BeamSearch , string , string , SpecialSymbols)

cdef class PyDecoder:
    cdef Decoder *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, char* wfstFile,char* labelsFile,int beamWidth,double pathAcceptingThreshold=0.):
        self.thisptr = new Decoder(wfstFile, labelsFile, beamWidth, pathAcceptingThreshold)

    def __dealloc__(self):
        del self.thisptr

    def decode(self,pyMat,lmWeight):
        #prepare input
        cdef vector[vector[double]] cMat = vector[vector[double]]()
        for row in pyMat:
            cMat.push_back(vector[double]())
            for col in row:
                cMat.back().push_back(col)

        # call function
        cdef vector[vector[string]] cVec = self.thisptr.decode(cMat,lmWeight)

        # prepare output
        cdef vector[vector[string]].iterator it = cVec.begin()
        pyVec = []
        while it != cVec.end():
            pyVec.append((deref(it)[0],deref(it)[1]))
            inc(it)

        #return output
        return pyVec