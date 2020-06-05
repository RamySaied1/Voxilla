from flask import Blueprint, jsonify, request, current_app,render_template, request
from flask_cors import cross_origin,CORS
upload = Blueprint('upload', __name__)
#CORS(upload)

############## maded only for test purposes it will removed after integration with front end
#@upload.route('/getfile')
def upload_file_gui():
   return render_template('upload.html')
	

#@upload.route('/', methods = ['POST'])
def upload_file():
   try:
      f = request.files['file']
      f.save(current_app.config["UPLOAD_FILE"])
      return jsonify({ "message": "file uploaded sussessfully" }) ,200
   except Exception as e:
      return jsonify({ "message": e.message }) ,404


