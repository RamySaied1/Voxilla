from config import Config
if Config.MOCK:
    asr_object = None
else:
    from app.recognition.asr import ASR
    
    model_arch=Config.RECOGNITION_DIR+'Classifier/model_cpu.json'
    model_weights=Config.RECOGNITION_DIR+'Classifier/weights.h5'
    model_priori_proba_file = Config.RECOGNITION_DIR+'Classifier/priori.txt'
    fst_folder=Config.RECOGNITION_DIR+'Decoder/Graphs/200k-vocab/'
    acoustic_model_labels_file = Config.RECOGNITION_DIR+'Decoder/Graphs/200k-vocab/labels.ciphones'
    words_lexicon_file = Config.RECOGNITION_DIR+'ForcedAlignmnet/words_lexicon.txt'
    phones_lexicon_file = Config.RECOGNITION_DIR+'ForcedAlignmnet/phones_lexicon.txt'
    asr_object = ASR(model_arch,model_weights, model_priori_proba_file, fst_folder,acoustic_model_labels_file,words_lexicon_file,phones_lexicon_file)
