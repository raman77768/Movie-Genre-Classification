from flask import Flask

UPLOAD_FOLDER = 'C:/Users/Raman/Desktop/movieproject'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER