from app import currentApp

if __name__ == '__main__':
    currentApp.run(debug=currentApp.config["DEBUG"],use_reloader=False)