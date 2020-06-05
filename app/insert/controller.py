from flask import Blueprint, jsonify, request, current_app,render_template
from flask_cors import cross_origin,CORS
from uuid import uuid4
import os
import librosa # TODO: use light library instead of librosa
import numpy as np
from app.insert import insert_object
insert = Blueprint('insert', __name__)
#CORS(insert)

#@insert.route('/getfile')
def upload_file_gui():
   return render_template('insert.html')
	
#@insert.route('/', methods = ['POST'])
def generate_word():
   try:
      text=request.json['text']["word"]
      #text=text.split()
      
      generated_wav,samplingRate=insert_object.getWav(current_app.config["UPLOAD_FILE"],[text])
      #print(samplingRate)
      # librosa.output.write_wav(current_app.config["DOWNLOAD_FILE"], generated_wav.astype(np.float32), samplingRate)
      fname = uuid4()
      os.path.join(current_app.config["DOWNLOAD_FILE"], fname)
      librosa.output.write_wav(current_app.config["DOWNLOAD_FILE"], generated_wav.astype(np.float32), samplingRate)
      return jsonify({ "fname": fname, "text": 'route accessed successfly' }) ,200
   except Exception as e:
      return jsonify({ "text": str(e) }) ,400
