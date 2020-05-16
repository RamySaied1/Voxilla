from flask import Blueprint, jsonify, request, current_app,render_template, request
from werkzeug import secure_filename

upload = Blueprint('upload', __name__)


############## maded only for test purposes it will removed after integration with front end
@upload.route('/getfile')
def upload_file_gui():
   return render_template('upload.html')
	

@upload.route('/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(current_app.config["UPLOAD_FILE"])
      return jsonify({ "message": "file uploaded sussessfully" }) ,200
   else:
      return jsonify({ "message": "method must be post" }) ,404


