from flask import Flask
from flask_cors import CORS
# Initialize the app
app = Flask(__name__, instance_relative_config=True)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Load the views
from app import views
from app import serve_static

# Load the config file
app.config.from_object('config')
