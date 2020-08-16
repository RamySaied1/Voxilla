import os
import librosa # TODO: use light library instead of librosa

from flask import jsonify, request
from hashlib import sha1 as sha

from .app import app, db
from .models import Clip
from .wrappers import userRequiredJson
from .recognition import asr_object
from .insert import insert_object
from .utils import createFileDirectory, randomFileName

from .recognition.common.common import frames_to_seconds
from .insert.insert import amplifySound

@app.route('/clips/<int:clipId>/recognize', methods=['GET'])
@userRequiredJson()
def recognizeWords(user, clipId): # speech recognition
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if(clip.project.user_id != userId):
        return jsonify({"err": "you can't make action on others clips"}), 401
    request.args.get("type")
    if(clip.fileUri == None):
        return jsonify({"err": "no file exist, upload first"}), 400
    print(clip.fileUri)

    max_active_tokens = app.config["MAX_ACTIVE_TOKENS"]
    beam_width = app.config["BEAM_WIDTH"]
    acoustic_model_weigh = app.config["ACOUSTIC_MODEL_WEIGHT"]

    frame_duration = app.config["FRAME_DURATION"]
    frame_shift = app.config["FRAME_SHIFT"]
    try:
        alignments = asr_object.speech_to_text(clip.fileUri, max_active_tokens, beam_width, acoustic_model_weigh, latticeBeam=4, include_alignment = True)
        return jsonify({ "text": frames_to_seconds(alignments, frame_duration, frame_shift) }), 200
    except  Exception as ex:
        return jsonify({ "Error": ex }), 400

@app.route('/clips/<int:clipId>/synthesize', methods=['POST'])
@userRequiredJson()
def synthesizeWords(user, clipId): # speech recognition
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if(clip.project.user_id != userId):
        return jsonify({"err": "you can't make action on others clips"}), 401
    if(clip.fileUri == None):
        return jsonify({"err": "no file exist, upload first"}), 400
    print(clip.fileUri)

    try:
        text = request.json['text']["word"]
        outFileName = sha(text.encode()).hexdigest() + ".wav"
        outPath = os.path.join(f"audios/{userId}/{clip.id}/synthesized", outFileName)

        generated_wav,samplingRate = insert_object.getWav(clip.fileUri, [text])
        generated_wav = amplifySound(generated_wav, clip.fileUri)
        librosa.output.write_wav(outPath, generated_wav.astype(np.float32), samplingRate)
        return jsonify({ "uri": outPath, "text": 'route accessed successfly' }) ,200
    except Exception as e:
        return jsonify({ "text": str(e) }), 400
