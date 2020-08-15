from datetime import datetime

from .app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    registrationDate = db.Column(db.DateTime, default=datetime.now)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(20), nullable=False, unique=True)
    description = db.Column(db.String(100), nullable=True)
    creationDate = db.Column(db.DateTime, default=datetime.now)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship("User", backref="projects")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Clip(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(20), nullable=False, unique=True)
    description = db.Column(db.String(100), nullable=True)
    creationDate = db.Column(db.DateTime, default=datetime.now)
    alignments = db.Column(db.Text, nullable=True)
    fileUri = db.Column(db.String(150), nullable=True)

    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), primary_key=True)
    project = db.relationship("Project", backref="clips")
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

