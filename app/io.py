import os

from flask import jsonify, request, send_file

from .app import app, db
from .models import Clip
from .wrappers import userRequiredJson

@app.route('/clips/<int:clipId>/upload', methods = ['POST'])
@userRequiredJson()
def uploadClipInfo(user, clipId):
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if clip.project.user_id != userId:
        return jsonify({"err": "can't edit others clips"}), 401
    
    try:
        f = request.files["file"]
        os.makedirs(f"audios/{userId}/", exist_ok=True)
        f.save(f"audios/{userId}/{clipId}.flac")
        clip.fileUri = f"audios/{userId}/{clipId}.flac"
        db.session.add(clip)
        db.session.commit()
        # f.save(app.config["UPLOAD_FILE"])
        return jsonify({ "message": "file uploaded sussessfully" }), 200
    except Exception as e:
        return jsonify({ "message": e.message }), 404

@app.route('/clips/<int:clipId>/download')
@userRequiredJson()
def downloadClipInfo(user, clipId):
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if clip.project.user_id != userId:
        return jsonify({"err": "can't download others clips"}), 401
    
    return send_file(f"..\\audios\\{userId}\\{clip.id}.flac", as_attachment=True)
