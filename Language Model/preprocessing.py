
#tokens that will be used to tokenize the sentences
SOS="<s> "
EOS="</s>"
UNK="<UNK>"

#add 
def addSentenceTokens(sentences,n):
    #sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(SOS, s, EOS) for s in sentences]
def countWords(sentences,vocab):
    countDict={token:0 for token in sentences }
    for word in sentences:
        if word in vocab:
            countDict[word]+=1
    return countDict



def replace_singletons(tokens,refVocab):
    """Replace tokens which appear only once in the corpus with <UNK>.
    
    Args:
        tokens (list of str): the tokens comprising the corpus.
    Returns:
        The same list of tokens with each singleton replaced by <UNK>.
    
    """
    vocab = countWords(tokens,refVocab)
    return [token if vocab[token] > 1 else UNK for token in tokens]

def preprocess(sentences,vocab,n):
    sentences = addSentenceTokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens=replace_singletons(tokens,vocab)
    return tokens
