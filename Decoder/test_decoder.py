import sys
sys.path.append("../FeatureExtraction/")
sys.path.append("../Classifier/")
sys.path.append("./python/")
sys.path.append("./c++/")
from os import path
import argparse
import collections
import re
from time import time
from fst import FST
from classifier import Classifier
from beam_search import *
from htk_featio import read_htk_user_feat
# from decoder_wrapper import PyDecoder as Decoder
from itertools import groupby


def parse_script_line(script_line: str, script_path: str):
    """
    This function parses utterance specifications from a script file.
    It is expected the format obeys the following structure: (utterance name).feat=(feature filename)[(start frame), (end_frame)]
    Returns:
    """
    m = re.match(r'(.*)\.feat=(.*)\[(\d+),(\d+)\]', script_line)
    assert(m)
    audio_features_filename = m.group(1)
    audio_features_filepath = m.group(2)
    frame_start = int(m.group(3))
    frame_end = int(m.group(4))
    m = re.match(r'\.\.\.[/\\](.*)', audio_features_filepath)
    if m:
        audio_features_filepath = path.join(script_path, m.group(1))
    return audio_features_filename, audio_features_filepath, frame_start, frame_end


def get_features(script_line: str, script_path: str):
    """
    This function extracts an utterance's string identifier and features from an archive.
    Returns:
        feat: An array representing the acoustic data, one row per frame of data.
        utt: A string identifier for the utterance.
    """
    audio_features_filename, audio_features_filepath, frame_start, frame_end = parse_script_line(
        script_line, script_path)
    feature_vectors = read_htk_user_feat(audio_features_filepath)
    assert(frame_start == 0)
    assert(frame_end + 1 - frame_start == len(feature_vectors))
    return feature_vectors, audio_features_filename

def write_ref_output(script_path, scirpt_num, ref_file="./ref.txt"):
    parts = scirpt_num.split('-')
    txtFileName = '-'.join(parts[:-1])
    txtFilePath = path.join(*(script_path,*parts[:-1], txtFileName+".trans.txt"))
    lines = []
    with open(txtFilePath,"r") as f:
        lines.append(f.readlines()[int(parts[-1])].split(" ",1)[1].lower())
    with open(ref_file,"a") as f:
        f.write(''.join(lines))
        

# python3 test_decoder.py -model ../Experiments/BLSTM1/BLSTM_CE -decoding_graph DecodingGraph.txt -input_labels ../Experiments/am/labels.ciphones -scr ../Experiments/lists/feat_test.rscp -outfile out2

def main():
    parser = argparse.ArgumentParser( description="Decode speech from parameter files.")
    parser.add_argument('-model', '--model', help='classifier file', required=True, default=None)
    parser.add_argument('-decoding_graph', '--decoding_graph', help="Text-format openfst decoding graph", required=True, default=None)
    parser.add_argument('-input_labels', '--input_labels', help="Text files containing input labels", required=True, default=None)
    parser.add_argument('-scr', '--script', help='script file  containing features files pathes to test on', required=True, default=None)
    parser.add_argument('-outfile', '--outfile', help='Filename to write output hypotheses',type=argparse.FileType('w'), required=True, default=None)
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score', required=False, type=float, default=30)
    parser.add_argument('-beam_width', '--beam_width', help='Maximum token count per frame', required=False, type=int, default=500)

    args = parser.parse_args()

    # load classifier
    classifier = Classifier(args.model)

    # load FST
    # decoder = Decoder(args.decoding_graph.encode(),args.input_labels.encode(),args.beam_width)

    
    all_time_start = 0
    predictedSents = []
    predictedAudioFileNames = []
    script_path = path.split(args.script)[0]
    try:
        with open(args.script, 'r') as ftest:
            all_time_start = time()
            for t,line in enumerate(ftest):
                time_start = time()
                feature_vectors, audio_features_filename = get_features(line.rstrip(), script_path)
                activations = classifier.eval(feature_vectors,do_stack_features=False) # returns all activations as shape (1, number of examples, number of last layer neurorns) we us [0] to remove the first dimention
                with open(f"{audio_features_filename}_BLSTM1_activations.txt","w") as f:
                    f.write(f"{activations.shape[0]} {activations.shape[1]}\n")
                    np.savetxt(f,activations)
                continue      
                # hypothesis,best_token = fst.decode(BeamSearch(args.beam_width), activations, lmweight=args.lmweight)
                hypothesis = decoder.decode(activations,30.)
                # print(hypothesis)
                # return
                hypothesis = [key for key, group in groupby(hypothesis)] # remove self loops
                predictedSents.append(' '.join([outlabel for _,outlabel in hypothesis if outlabel not in ["<eps>","<s>","</s>"]]))
                print(f"line {line} takes {int(time()-time_start)} seconds")
                write_ref_output("../LibriSpeech/dev-clean",audio_features_filename)
    except KeyboardInterrupt:
        print("[CTRL+C detected]")
        text = '\n'.join(predictedSents)
        args.outfile.write(text)  # remove firlst endline

    text = '\n'.join(predictedSents)
    args.outfile.write(text)  # remove firlst endline
    print(f"total time  takes {int(time()-all_time_start)} seconds")


if __name__ == '__main__':
    main()
