import os
from uuid import uuid4




def createFileDirectory(filePath, exist_ok = True):
    directory = os.path.dirname(filePath)
    os.makedirs(directory, exist_ok=exist_ok)


def randomFileName(ext=".wav"):
    return str(uuid4()) + ext