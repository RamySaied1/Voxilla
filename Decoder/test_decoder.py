import sys
sys.path.append("../FeatureExtraction/")
sys.path.append("../Classifier/")
sys.path.append("./python/")
import os.path
import argparse
import collections
import re
from time import time
from fst import FST
from classifier import Classifier
from beam_search import *
from htk_featio import read_htk_user_feat


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
        audio_features_filepath = os.path.join(script_path, m.group(1))
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

# python test_decoder.py -model ../Experiments/DNN/DNN_CE -decoding_graph DecodingGraph.txt -input_labels ../Experiments/am/labels.ciphones -scr ../Experiments/lists/feat_dev.rscp -outfile out2

def main():
    parser = argparse.ArgumentParser( description="Decode speech from parameter files.")
    parser.add_argument('-model', '--model', help='classifier file', required=True, default=None)
    parser.add_argument('-decoding_graph', '--decoding_graph', help="Text-format openfst decoding graph", required=True, default=None)
    parser.add_argument('-input_labels', '--input_labels', help="Text files containing input labels", required=True, default=None)
    parser.add_argument('-scr', '--script', help='script file  containing features files pathes to test on', required=True, default=None)
    parser.add_argument('-outfile', '--outfile', help='Filename to write output hypotheses', required=True, default=None)
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score', required=False, type=float, default=30)
    parser.add_argument('-beam_width', '--beam_width', help='Maximum token count per frame', required=False, type=int, default=500)

    args = parser.parse_args()

    # load classifier
    classifier = Classifier(args.model)

    # load FST
    fst = FST(args.decoding_graph, args.input_labels)
    
    all_time_start = 0
    predictedSents = []
    script_path = os.path.split(args.script)[0]
    try:
        with open(args.script, 'r') as ftest:
            with open(args.outfile, 'w', buffering=1) as fout:
                all_time_start = time()
                for line in ftest:
                    time_start = time()
                    feature_vectors, audio_features_filename = get_features(line.rstrip(), script_path)
                    activations = classifier.eval(feature_vectors,do_stack_features=True) # returns all activations as shape (1, number of examples, number of last layer neurorns) we us [0] to remove the first dimention
                    hypothesis,best_token = fst.decode(BeamSearch(args.beam_width), activations, lmweight=args.lmweight)
                    predictedSents.append(' '.join([outlabel for _,outlabel,_ in hypothesis if outlabel not in fst.speial_syms]))

                    print("best cost: AM={} LM={} JOINT={}".format(best_token.model_score, best_token.lm_score, best_token.model_score + best_token.lm_score))
                    print(f"line {line} takes {int(time()-time_start)} seconds")
    except KeyboardInterrupt:
        print("[CTRL+C detected]")
        text = '\n'.join(predictedSents)
        args.outfile.write(text)  # remove firlst endline

    text = '\n'.join(predictedSents)
    args.outfile.write(text)  # remove firlst endline
    print(f"total time  takes {int(time()-all_time_start)} seconds")


if __name__ == '__main__':
    main()
