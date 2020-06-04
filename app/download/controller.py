from flask import Blueprint, jsonify, request, current_app,render_template, request,send_file


download = Blueprint('download', __name__)
	

@download.route('/', methods = ['GET','POST'])
def download_file():
   try:
      print(current_app.config["DOWNLOAD_FILE"])
      return send_file(current_app.config["DOWNLOAD_FILE"], as_attachment=True)
   except Exception as e:
      return jsonify({ "message": e.message }) ,404


