import os.path
import argparse
import collections
import re
import sys; 
from time import time
from asr import ASR

# python3 test_asr.py -model ./Experiments/BLSTM1/BLSTM_CE -fst_folder ./Decoder/Graphs/200k-vocab/ -acoustic_model_labels_file ./Decoder/Graphs/200k-vocab/labels.ciphones -srcfile waves_test.txt -outfile transcript

def main():
    parser = argparse.ArgumentParser( description="Decode speech from parameter files.")
    parser.add_argument('-model', '--model', help='classifier file', required=True, default=None)
    parser.add_argument('-fst_folder', '--fst_folder', help="folder containg HCLG graph", required=True, default=None)
    parser.add_argument('-acoustic_model_labels_file', '--acoustic_model_labels_file', help="Text file name containing input labels", required=True, default=None)
    parser.add_argument('-srcfile', '--srcfile', help='file containing wave files pathes to test on', required=True, default=None)
    parser.add_argument('-outfile', '--outfile', help='Filename to write output hypotheses', required=True, default=None)
    parser.add_argument('-amw', '--amw', help='Relative weight of LM score', required=False, type=float, default=9)
    parser.add_argument('-max_active_tokens', '--max_active_tokens', help='Maximum token count per frame', required=False, type=int, default=1000)
    parser.add_argument('-beam_width', '--beam_width', help='Maximum token count per frame', required=False, type=float, default=9.0)

    args = parser.parse_args()

    asr = ASR(args.model, args.fst_folder, args.acoustic_model_labels_file)

    all_time_start = time()
    predictedSents = []
    try:
        with open(args.srcfile, 'r') as ftest:
            with open(args.outfile, 'w', buffering=1) as fout:
                all_time_start = time()
                for line in ftest.read().split():
                    text = asr.speech_to_text(line,1000,12,3,include_alignment=False)
                    predictedSents.append(text)
    except KeyboardInterrupt:
        print("[CTRL+C detected]")
        text = '\n'.join(predictedSents)
        with open(args.outfile,"w") as f:
            f.write(text)

    text = '\n'.join(predictedSents)
    with open(args.outfile,"w") as f:
        f.write(text)
    print(f"total time  takes {int(time()-all_time_start)} seconds")


if __name__ == '__main__':
    main()
