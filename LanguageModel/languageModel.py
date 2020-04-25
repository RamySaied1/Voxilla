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
    return vocab
def writeOutput(outputDict,num):
    outputFile=open("out.txt",'a')
    outputFile.write(f"{num}-grams:\n")
    for key,value in outputDict.items():
        #print(f"writing {key} : {value}")
        if value==0:
            print(key)
        count=log10(value["count"])
        backOff=log10(value["backoff"])
        outputFile.write(f"{count}  {key} {backOff} \n")
        
    outputFile.close()    

class LanguageModel:
    def __init__(self,dataDir,vocabDir,n,laplace=1):
        self.n=n
        self.vocab=loadVocab(vocabDir)
        self.tokens=preprocess(loadData(dataDir),self.vocab,self.n)
        self.countDict,self.wordsCount=countWords(self.tokens,self.vocab)
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
        nGrams=self.getNGrams(num)
        nGramsTokens=list(itertools.permutations(self.vocab,num))
        nGramsStatisitcs,_=countWords(nGrams,nGramsTokens)
        nGramsStatisitcs={nGram:val for nGram,val in nGramsStatisitcs.items() if val>100}
        print("a")
        
        mGrams = self.getNGrams(num-1)
        mGramsTokens=list(itertools.permutations(self.vocab,num-1))
        mGramsStatisitcs,_ = countWords(mGrams,mGramsTokens)
        mGramsStatisitcs={mGram:val for mGram,val in mGramsStatisitcs.items() if val>100}
        print("b")
        
        def smoothed_count(nGram, nCount):
            mGram = nGram[:-1]
            mCount = mGramsStatisitcs[mGram]
            
            return (nCount) / (mCount + self.laplace * self.countEvents(mGram,nGramsStatisitcs.keys()))

        return { nGram: {"count":smoothed_count(nGram, count),"backoff":None} for nGram, count in nGramsStatisitcs.items() }
    
    def countEvents(self,token,vocab):
        count=0
        tokenLen=len(token)
        for v in vocab:
            if token == v[0:tokenLen]:
                count+=1
        return count

    def createModel(self):
        
        #unigrams are a special case 
        num_tokens=len(self.tokens)
        print("Counting Unigrams")
        unigrams={(unigrams,): {"count":count/ (num_tokens+ self.wordsCount),"backoff":None}  for unigrams, count in self.countDict.items()}
        print("Unigrams is Done,Counting bigrams")
        
        bigrams=self.smooth(2)
        print("bigrams is Done,counting trigrams")
        
        trigrams=self.smooth(3)
        print("trigrams is Done")
        
        
        #calculating backoff weights 
        for key,val in unigrams.items():
            numEvents=self.countEvents(key,bigrams)
            if numEvents==0:
               val["backoff"]=""
            else:
                val["backoff"]=numEvents/(numEvents+val["count"])
        
        for key,val in bigrams.items():
            numEvents=self.countEvents(key,trigrams)
            if numEvents==0:
               val["backoff"]=""
            else:
                val["backoff"]=numEvents/(numEvents+val["count"])
        
        #print(unigrams)
        writeOutput(unigrams,1)
        writeOutput(bigrams,2)
        writeOutput(trigrams,3)

        


if __name__=="__main__":
    lm=LanguageModel("test.txt","librispeech.top10k.1grams",3,1)
    lm.createModel()