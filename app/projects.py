from flask import jsonify, request

from .app import app, db
from .models import User, Project
from .wrappers import userRequiredJson

@app.route("/projects", methods=['GET'])
@userRequiredJson()
def getProjects(user):
    userId = user["id"]
    user = User.query.filter_by(id=userId).first()
    projects = [p.as_dict() for p in user.projects]
    print(projects)
    return jsonify({"projects": projects}), 200

@app.route("/projects/<int:projectId>", methods=['DELETE'])
@userRequiredJson()
def deleteProject(user, projectId):
    userId = user["id"]
    project = Project.query.filter_by(id=projectId).first()
    if (project.user_id != userId):
        return jsonify({"err": "you can't delete others projects"}), 401
    Project.query.filter_by(id=projectId).delete()
    db.session.commit()
    return jsonify({"project": project.as_dict()}), 200

@app.route("/projects/<projectId>", methods=['PUT'])
@userRequiredJson()
def editProject(user, projectId):
    userId = user["id"]
    project = Project.query.filter_by(id=projectId).first()
    if (project.user_id != userId):
        return jsonify({"err": "you can't edit others projects"}), 401
    data_json = request.get_json()
    print(data_json)
    project.description = data_json["desc"]
    project.title = data_json["title"]
    db.session.add(project)
    db.session.commit()
    return jsonify({"project": project.as_dict()}), 200

@app.route("/projects", methods=['POST'])
@userRequiredJson()
def addProject(user):
    userId = user["id"]
    data_json = request.get_json()
    newProject = Project(title=data_json["title"], description=data_json["desc"], user_id=userId)
    db.session.add(newProject)
    db.session.commit()
    return jsonify({"project": newProject.as_dict()}), 200
