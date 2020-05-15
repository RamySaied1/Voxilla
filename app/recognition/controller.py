from flask import Blueprint, jsonify, request, current_app,render_template, request
from werkzeug import secure_filename

from app.recognition import asr_object


recognition = Blueprint('recognition', __name__)

@recognition.route('/upload')
def upload():
   return render_template('upload.html')
	
@recognition.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(current_app.config["UPLOAD_FILE"])
      return 'file uploaded successfully'

@recognition.route('transcript/', methods=['GET'])
def getTranscript():
   wav_file=current_app.config["UPLOAD_FILE"]
   max_active_tokens=current_app.config["MAX_ACTIVE_TOKENS"]
   beam_width=current_app.config["BEAM_WIDTH"]
   acoustic_model_weigh=current_app.config["ACOUSTIC_MODEL_WEIGHT"]
   text=""
   #try:
   text=asr_object.speech_to_text(wav_file, max_active_tokens, beam_width, acoustic_model_weigh)
   #except  Exception as ex:
   #   return jsonify({ "Error": ex.message }) ,404
   return jsonify({ "text": text }) ,200


