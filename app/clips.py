from flask import jsonify, request

from .app import app, db
from .models import Project, Clip
from .wrappers import userRequiredJson

@app.route("/<int:projectId>/clips")
@userRequiredJson()
def getClips(user, projectId):
    project = Project.query.filter_by(id=projectId).first()
    if (project.user_id != user["id"]):
        return jsonify({"err": "you can't access others clips"}), 401
    clips = [c.as_dict() for c in project.clips]
    print(clips)
    return jsonify({"project": project.as_dict(), "clips": clips}), 200

@app.route("/<int:projId>/clips", methods=['POST'])
@userRequiredJson()
def addClip(user, projId):
    userId = user["id"]
    data_json = request.get_json()
    project = Project.query.filter_by(id=projId).first()
    if project.user_id != userId:
        return jsonify({"err": "you can't add to others projects"}), 401
    newClip = Clip(title=data_json["title"], description=data_json["desc"], project_id=projId)
    db.session.add(newClip)
    db.session.commit()
    return jsonify({"clip": newClip.as_dict()}), 200

@app.route("/clips/<clipId>", methods=['PUT'])
@userRequiredJson()
def editClip(user, clipId):
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if (clip.project.user_id != userId):
        return jsonify({"err": "you can't edit others clips"}), 401
    data_json = request.get_json()
    print(data_json)
    clip.description = data_json["desc"]
    clip.title = data_json["title"]
    db.session.add(clip)
    db.session.commit()
    return jsonify({"clip": clip.as_dict()}), 200

@app.route("/clips/<int:clipId>", methods=['DELETE'])
@userRequiredJson()
def deleteClip(user, clipId):
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if (clip.project.user_id != userId):
        return jsonify({"err": "you can't edit others clips"}), 401
    Clip.query.filter_by(id=clipId).delete()
    db.session.commit()
    return jsonify({"clip": clip.as_dict()}), 200

@app.route("/<int:projectId>/<int:clipId>")
@userRequiredJson()
def downloadClip(user, projectId, clipId):
    project = Project.query.filter_by(id=projectId).first()
    project_dict = project.as_dict()
    if (project_dict.user_id != user["id"]):
        return jsonify({"err": "you can't access others clips"}), 401
    clips = [c.as_dict() for c in project.clips]
    print(clips)
    return jsonify({"project": project_dict, "clips": clips}), 200
