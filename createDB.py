from werkzeug.security import generate_password_hash

from app import db
from app.models import User, Project, Clip

def multiAdd(*values):
	for v in values:
		db.session.add(v)
	db.session.commit()

db.drop_all()
db.create_all()

admin = User(id=0, email='admin@gmail.com', password='admin@password.com', username='Root')
db.session.add(admin)
db.session.commit()

hashedPassword = generate_password_hash("123456789", method='sha256')
u1 = User(id=1,email='user1@gmail.com', password=hashedPassword, username="user1")
u2 = User(id=2,email='user2@gmail.com', password=hashedPassword, username="user2")
u3 = User(id=3,email='user3@gmail.com', password=hashedPassword, username="user3")
u4 = User(id=4,email='user4@gmail.com', password=hashedPassword, username="user4")
multiAdd(u1,u2,u3,u4)

p1 = Project(id=1, title="p1", description="project-1", user=u1)
p2 = Project(id=2, title="p2", description="project-2", user=u2)
p3 = Project(id=3, title="p3", description="project-3", user=u3)
p4 = Project(id=4, title="p4", description="project-4", user=u4)
multiAdd(p1, p2, p3, p4)

c1 = Clip(id=1, title="c1", project=p1)
c2 = Clip(id=2, title="c2", project=p2)
c3 = Clip(id=3, title="c3", fileUri="clips/clip3.flac", project=p3) # AmmarMachine:
c4 = Clip(id=4, title="c4", fileUri="clips/clip4.flac", project=p4) # AmmarMachine: 
multiAdd(c1, c2, c3, c4)

db.session.commit()
