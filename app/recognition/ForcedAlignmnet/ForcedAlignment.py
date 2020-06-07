from AcousticsSegmentTree import AcousticsSegmentTree as AST
from itertools import groupby
import string
import numpy as np
import re


class ForcedAlignment():
    def __init__(self,wordsLexiconFile = "words_lexicon.txt", phonesLexiconFile = "phones_lexicon.txt", silence_phones = ["sil","spn"]):
        self.wordsLexicon = {}
        self.phonesLexicon = {}
        self.silWords = {}
        self.silence_phones = silence_phones
        self.lexicons = [self.wordsLexicon,self.phonesLexicon]
        lexiconFiles = [wordsLexiconFile ,phonesLexiconFile]

        for lexicon,lexiconFile in zip(self.lexicons,lexiconFiles):
            with open(lexiconFile,"r") as lf:
                for line in lf:
                    tokens = line.split()
                    lexicon[tokens[0]] = lexicon.get(tokens[0],[])
                    lexicon[tokens[0]].append(tokens[1:])
                    if(lexiconFile == wordsLexiconFile and tokens[1] in self.silence_phones):
                        self.silWords[tokens[1]] = tokens[0] # ingore <spoken_noise> word and use only <unk> it will be overriden due to order of file

        
        # define some useful functions
        self.sub_phone_to_phone = lambda sph: re.sub("_s[0-9]","",sph)
        self.is_sil_phone = lambda ph: ph in self.silence_phones
        self.is_sil_subphone = lambda sph: self.is_sil_phone(sph.split("_")[0])

    def _preprocess(self,decoderOutPath):
        # uniquePairs = np.array([key for key, group in groupby(decoderOutPath)])
        # ensure all strings are lower case
        tolower = np.vectorize(lambda x: x.lower() ,otypes=[str])
        decoderOutPath = tolower(np.array(decoderOutPath))
        
        def process_silence(subPhones):
            def transform_ifsil(subPhone):
                if(self.is_sil_subphone(subPhone)): return self.sub_phone_to_phone(subPhone)+"_s2"
                return subPhone
            return np.vectorize(transform_ifsil)(subPhones)
        
        #process silence phones
        decoderOutPath[:,0] = process_silence(decoderOutPath[:,0]) 

        #remove <eps> input
        maskRows = np.apply_along_axis(lambda inoutid: not(inoutid[0]=="<eps>"),axis=1,arr=decoderOutPath)
        inpLabels = decoderOutPath[maskRows][:,0]

        #replace <unk> from with <eps>
        replaceUnk = np.vectorize(lambda x: "<eps>" if x=="<unk>" else x,otypes=[str])
        decoderOutPath[:,1] = replaceUnk(decoderOutPath[:,1])


        predicate = lambda inoutid: not(inoutid[0]==inoutid[1]=="<eps>")
        maskRows = np.apply_along_axis(predicate,axis=1,arr=decoderOutPath)
        uniquePairs = [key for key, group in groupby(decoderOutPath[maskRows], lambda x : ",".join(x[:2]))]
        outLabels = [inout.split(",")[1] for inout in uniquePairs]
        outLabels = list(filter(lambda w : w not in  ["<eps>","<s>","</s>"],outLabels))

        # override silence phone lexicon
        for phone in self.silence_phones:
            self.phonesLexicon[phone] = [[phone + "_s2"]]

        return inpLabels,outLabels

    def _get_groupings(self,parents,childeren,associations,childerenGroupings,parentI=0,childI=0):
        if(childI >= len(childeren)):
            return True
        
        silWords = list(self.silWords.values())
        possibleParents = silWords + [parents[parentI]] if parentI < len(parents) else silWords
        for possibleParent in possibleParents: # always try silent as possible parent and insert it if it's right
            for assoc in associations[possibleParent]:
                if(childI + len(assoc) <= len(childeren) and childeren[childI:childI+len(assoc)] == assoc):
                    childerenGroupings.append((len(assoc),possibleParent))
                    nextParentI = parentI+1 if possibleParent not in silWords else parentI
                    if(self._get_groupings(parents,childeren,associations,childerenGroupings,nextParentI,childI+len(assoc))):
                        if(possibleParent in silWords):
                            parents.insert(parentI, possibleParent)
                        return True
                    childerenGroupings.pop()
        return False

    def align(self,decoderOutPath,framesCount):
        inpLabels,outLabels = self._preprocess(decoderOutPath)
        assert(framesCount == len(inpLabels))

        # group repeated frames
        framesGroupings = [(len(list(group)),subPhone) for subPhone, group in groupby(inpLabels)]
        
        # group subPhones to phones
        subPhones = [subPhone for _,subPhone in framesGroupings if "2" in subPhone]
        phones = [self.sub_phone_to_phone(subPhone) for subPhone in subPhones]
        subPhoneGroupings = [(len(self.phonesLexicon[phone][0]),phone) for phone in phones]

        # group phones to words
        phonesGroupings=[]
        assert(self._get_groupings(outLabels,phones,self.wordsLexicon,phonesGroupings))
        
        # group words to sents
        wordsGroupings = [(len(outLabels),"SENT")]

        # levels groups
        levelsGroups = [wordsGroupings,phonesGroupings,subPhoneGroupings,framesGroupings]
        
        acoustics_seg_tree = AST()
        acoustics_seg_tree.build(levelsGroups,["sentences","words","phones","subphones","frames"])
        return acoustics_seg_tree
