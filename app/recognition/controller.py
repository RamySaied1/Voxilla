from flask import Blueprint, jsonify, request, current_app,render_template, request
from flask_cors import cross_origin,CORS

from app.recognition import asr_object
from app.recognition.common.common import frames_to_seconds


recognition = Blueprint('recognition', __name__)
#CORS(recognition)

#@recognition.route('gettranscript/', methods=['GET'])
def getTranscript():
   wav_file=current_app.config["UPLOAD_FILE"]
   max_active_tokens=current_app.config["MAX_ACTIVE_TOKENS"]
   beam_width=current_app.config["BEAM_WIDTH"]
   acoustic_model_weigh=current_app.config["ACOUSTIC_MODEL_WEIGHT"]
   text=""
   try:
      text=asr_object.speech_to_text(wav_file, max_active_tokens, beam_width, acoustic_model_weigh)
      return jsonify({ "text": text }) ,200
   except  Exception as ex:
      return jsonify({ "Error": ex }) ,404
   
#@recognition.route('getalignment/', methods=['GET'])
def getAlignment():
   wav_file=current_app.config["UPLOAD_FILE"]
   max_active_tokens=current_app.config["MAX_ACTIVE_TOKENS"]
   beam_width=current_app.config["BEAM_WIDTH"]
   acoustic_model_weigh=current_app.config["ACOUSTIC_MODEL_WEIGHT"]

   frame_duration=current_app.config["FRAME_DURATION"]
   frame_shift=current_app.config["FRAME_SHIFT"]
   alignments=[]
   try:
      alignments=asr_object.speech_to_text(wav_file, max_active_tokens, beam_width, acoustic_model_weigh,latticeBeam=4,include_alignment = True)
      return jsonify({ "text": frames_to_seconds(alignments,frame_duration,frame_shift) }) ,200
   except  Exception as ex:
      return jsonify({ "Error": ex }) ,404


