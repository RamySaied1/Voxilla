from flask import Blueprint, jsonify, request, current_app,render_template

insert = Blueprint('insert', __name__)


############## maded only for test purposes it will be removed after integration with front end
@insert.route('/getfile')
def upload_file_gui():
   return render_template('insert.html')
	

@insert.route('/', methods = ['POST'])
def generate_word():
   try:
      text=request.form.get('text')
      return jsonify({ "text": 'route accessed successfly' }) ,200
   except Exception as e:
      return jsonify({ "text": 'bad request shape' }) ,400
