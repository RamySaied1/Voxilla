from math import log10
import itertools
from preprocessing import preprocess,countWords

def loadData(dataDir):
    with open(dataDir,'r') as f:
        train = [l.strip() for l in f.readlines()]
    return train
def loadVocab(vocabDir):
    with open(vocabDir,'r') as f:
        vocab=[l.split()[0] for l in f.readlines()]
    vocab.append("<UNK>")
    return vocab
def writeOutput(outputDict,num):
    outputFile=open("out.txt",'a')
    outputFile.write(f"{num}-grams:\n")
    for key,value in outputDict.items():
        #print(f"writing {key} : {value}")
        if value==0:
            print(key)
        outputFile.write(f"{log10(value)}  {key}\n")
    outputFile.close()    

class LanguageModel:
    def __init__(self,dataDir,vocabDir,n,laplace=1):
        self.n=n
        self.vocab=loadVocab(vocabDir)
        self.tokens=preprocess(loadData(dataDir),self.vocab,self.n)
        self.countDict=countWords(self.tokens,self.vocab)
        self.laplace=laplace

    
    def getNGrams(self,num):
        nGrams=[]
        for i in range(0,len(self.tokens)-num):
            #append n elements starting from i to i+n
            temp=[]
            for j in range(0,num):
                temp.append(self.tokens[i+j])
            nGrams.append(tuple(temp))
        return nGrams        
    
    #apply laplacian smoothing
    def smooth(self,num):
        vocab_size=len(self.countDict)
        nGrams=self.getNGrams(num)
        nGramsTokens=list(itertools.permutations(self.vocab,num))
        nGramsStatisitcs=countWords(nGrams,nGramsTokens)
        
        mGrams = self.getNGrams(num-1)
        mGramsTokens=list(itertools.permutations(self.vocab,num-1))
        mGramsStatisitcs = countWords(mGrams,mGramsTokens)

        def smoothed_count(nGram, nCount):
            mGram = nGram[:-1]
            mCount = mGramsStatisitcs[mGram]
            return (nCount + self.laplace) / (mCount + self.laplace * vocab_size)

        return { nGram: smoothed_count(nGram, count) for nGram, count in nGramsStatisitcs.items() }
    def createModel(self):
        
        #unigrams are a special case 
        num_tokens=len(self.tokens)
        unigram={(unigram,): count / (len(self.vocab)+ num_tokens) for unigram, count in self.countDict.items()}
        #print(unigram)
        writeOutput(unigram,1)
        for i in range(2,self.n+1):
            print(f"counting {i} grams")
            #every iteration we will make a gram i.e:unigram,bigram.....n_gram
            writeOutput(self.smooth(i),i)

        


if __name__=="__main__":
    lm=LanguageModel("ami-train.txt","librispeech.top10k.1grams",3,1)
    lm.createModel()