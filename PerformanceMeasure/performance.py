import numpy as np
import argparse

def sed(ref=None, hyp=None):
    if ref is None or hyp is None:
        RuntimeError("ref and hyp are required, cannot be None")

    if (len(hyp)==0):
        refLen = len(ref)
        return (refLen, refLen, refLen, 0, 0)

    memo = np.ones((len(ref)+1,len(hyp)+1),dtype=np.int32) * np.iinfo(np.int32).max
    memo[-1][-1] = 0

    for i in range(len(ref)):
        for j in range(len(hyp)):
            if(ref[i]==hyp[j]):
                memo[i][j] = memo[i-1][j-1]
            else:
                memo[i][j] = 1 + min(memo[i-1,j],memo[i,j-1],memo[i-1,j-1])
    
    i = len(ref)-1
    j = len(hyp)-1
    total_errs = memo[i,j]
    del_errs = ins_errs = sub_err = 0
    while(total_errs):
        if(ref[i] == hyp[j]):
            i-=1
            j-=1
        else:
            total_errs-=1
            if(memo[i-1,j] == total_errs):
                i-=1
                del_errs+=1
            elif(memo[i,j-1] == total_errs):
                j-=1
                ins_errs+=1
            elif(memo[i-1,j-1] == total_errs):
                i-=1
                j-=1
                sub_err+=1
            else:
                RuntimeError("unexpected behavoir in string edit distance")

    return (len(ref),ins_errs+del_errs+sub_err, ins_errs,del_errs,sub_err)



def string_edit_distance(ref=None, hyp=None):
    if ref is None or hyp is None:
        RuntimeError("ref and hyp are required, cannot be None")

    x = ref
    y = hyp
    tokens = len(x)
    if (len(hyp)==0):
        return (tokens, tokens, tokens, 0, 0)

    # p[ix,iy] consumed ix tokens from x, iy tokens from y
    p = np.PINF * np.ones((len(x) + 1, len(y) + 1)) # track total errors
    e = np.zeros((len(x)+1, len(y) + 1, 3), dtype=np.int) # track deletions, insertions, substitutions
    p[0] = 0
    for ix in range(len(x) + 1):
        for iy in range(len(y) + 1):
            cst = np.PINF*np.ones([3])
            s = 0
            if ix > 0:
                cst[0] = p[ix - 1, iy] + 1 # deletion cost
            if iy > 0:
                cst[1] = p[ix, iy - 1] + 1 # insertion cost
            if ix > 0 and iy > 0:
                s = (1 if x[ix - 1] != y[iy -1] else 0)
                cst[2] = p[ix - 1, iy - 1] + s # substitution cost
            if ix > 0 or iy > 0:
                idx = np.argmin(cst) # if tied, one that occurs first wins
                p[ix, iy] = cst[idx]

                if (idx==0): # deletion
                    e[ix, iy, :] = e[ix - 1, iy, :]
                    e[ix, iy, 0] += 1
                elif (idx==1): # insertion
                    e[ix, iy, :] = e[ix, iy - 1, :]
                    e[ix, iy, 1] += 1
                elif (idx==2): # substitution
                    e[ix, iy, :] = e[ix - 1, iy - 1, :]
                    e[ix, iy, 2] += s

    edits = int(p[-1,-1])
    deletions, insertions, substitutions = e[-1, -1, :]
    return (tokens, edits, deletions, insertions, substitutions)

def read_trn_hyp_files(ref_trn=None, hyp_trn=None):
    with open(ref_trn) as reference:
        with open(hyp_trn) as hypothesis:
            return  hypothesis.readlines(),reference.readlines()


def score(hypothesis : [str]=None, reference : [str]=None):
    assert(len(hypothesis) == len(reference))
    wec = sec = total_tokens_n = total_del = total_insert = total_sub = 0
    for reference_string, hypothesis_string in zip(reference,hypothesis):
        reference_string_words = reference_string.split()
        hypothesis_string_words = hypothesis_string.split()
        tokens_n, errs_n, del_n, insert_n, sub_n = string_edit_distance(ref=reference_string_words, hyp=hypothesis_string_words)
        wec += errs_n
        total_del+=del_n; total_insert+=insert_n; total_sub+=sub_n
        sec += 1 if errs_n>0 else 0
        total_tokens_n += tokens_n
        print(f'ref -> {reference_string}hyp -> {hypothesis_string}words_n: {tokens_n}, sub_errs: {sub_n}, del_errs: {del_n}, insert_errs: {insert_n}, total_erros: {errs_n} \n----------------')
    print(f'total del: {total_del}, total insert: {total_insert}, total sub: {total_sub}')
    wer = wec/total_tokens_n
    ser = sec/len(hypothesis)
    print(f'wer% {wec}/{total_tokens_n} : {round(wer*100,2)}, ser% {sec}/{len(hypothesis)} : {round(ser*100,2)}')
    return wer,ser




if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-hyp', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-ref', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(*read_trn_hyp_files(ref_trn=args.reftrn, hyp_trn=args.hyptrn))
