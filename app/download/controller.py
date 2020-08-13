from flask import Blueprint, jsonify, request, current_app,render_template, request,send_file
from flask_cors import cross_origin,CORS
import os

download = Blueprint('download', __name__)
#CORS(download)	

#@download.route('/', methods = ['GET','POST'])
def download_file(fname):
   try:
      # print(current_app.config["DOWNLOAD_FILE"])
      # return send_file(current_app.config["DOWNLOAD_FILE"], as_attachment=True)
      path = os.path.join(current_app.config["DOWNLOAD_FILE"], fname)
      return send_file(path, as_attachment=True)
   except Exception as e:
      return jsonify({ "message": e.message }) ,404


