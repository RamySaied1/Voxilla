import os.path
import argparse
import collections
import re
import sys
sys.path.append("./PerformanceMeasure/")
from time import time
from asr import ASR
from performance import *

# scr file , classifier
#python3 test_asr.py -model_arch ./Classifier/model_cpu.json  -model_weights ./Classifier/weights.h5 -fst_folder ./Decoder/Graphs/200k-vocab/ -acoustic_model_labels_file ./Decoder/Graphs/200k-vocab/labels.ciphones -srcfile waves_test.txt -outfile pred_transcript.txt -reffile ref_transcript.txt

def write_ref_output(wav_file,ref_file="ref_transcript.txt"):
    transFilePath = wav_file[::-1].split('-',1)[1][::-1] + ".trans.txt"
    with open(transFilePath,"r") as f:
        lines = f.read().splitlines()
        filename = text = ""
        for line in lines:
            filename, text = line.lower().split(" ",1)
            if(filename+".flac" == wav_file.split("/")[-1]): break
    with open(ref_file,"a") as f:
        f.write(text+"\n")

def main():
    parser = argparse.ArgumentParser( description="Decode speech from parameter files.")
    parser.add_argument('-model_arch', '--model_arch', help='classifier file arch', required=True, default=None)
    parser.add_argument('-model_weights', '--model_weights', help='classifier file weights', required=True, default=None)
    parser.add_argument('-model_priori_proba_file', '--model_priori_proba_file', help='activations priori proba file', required=True, default=None)
    parser.add_argument('-fst_folder', '--fst_folder', help="folder containg HCLG graph", required=True, default=None)
    parser.add_argument('-acoustic_model_labels_file', '--acoustic_model_labels_file', help="Text file name containing input labels", required=True, default=None)
    parser.add_argument('-srcfile', '--srcfile', help='file containing wave files pathes to test on', required=True, default=None)
    parser.add_argument('-outfile', '--outfile', help='Filename to write output hypotheses', required=True, default=None)
    parser.add_argument('-reffile', '--reffile', help='Filename to write ref hypotheses', required=True, default=None)
    parser.add_argument('-amw', '--amw', help='Relative weight of LM score', required=False, type=float, default=9)
    parser.add_argument('-max_active_tokens', '--max_active_tokens', help='Maximum token count per frame', required=False, type=int, default=1000)
    parser.add_argument('-beam_width', '--beam_width', help='Maximum token count per frame', required=False, type=float, default=12.0)

    args = parser.parse_args()

    asr = ASR(args.model_arch,args.model_weights, args.model_priori_proba_file, args.fst_folder, args.acoustic_model_labels_file)

    all_time_start = time()
    predictedSents = []
    with open(args.reffile,"w") as f:
        pass # clear file
    try:
        with open(args.srcfile, 'r') as ftest:
            with open(args.outfile, 'w', buffering=1) as fout:
                all_time_start = time()
                for line in ftest.read().split():
                    text = asr.speech_to_text(line,1000,12,3,include_alignment=False)
                    predictedSents.append(text)
                    write_ref_output(line)
    except KeyboardInterrupt:
        print("[CTRL+C detected]")
        text = '\n'.join(predictedSents)
        with open(args.outfile,"w") as f:
            f.write(text)

    text = '\n'.join(predictedSents)
    with open(args.outfile,"w") as f:
        f.write(text)
    print(f"total time  takes {int(time()-all_time_start)} seconds")

    # run performance measure
    score(*read_trn_hyp_files(ref_trn=args.reffile, hyp_trn=args.outfile))


if __name__ == '__main__':
    main()
