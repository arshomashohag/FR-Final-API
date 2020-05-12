import os
from pathlib import Path

from flask import send_from_directory

from app import app
from app.config import app_config


@app.route('/images/<path:id>')
def get_images(id):
    img_path = os.path.join(Path().resolve().cwd(), 'data/images/' + id + '.jpg')

    if os.path.exists(img_path):
        return send_from_directory('../' + app_config['DATA_IMAGE_ROOT'], id + '.jpg')

    return send_from_directory(app_config['STATIC_ROOT'], 'notfound.png')


@app.route('/banner/<path:name>')
def get_banners(name):
    static_path = os.path.join(app.root_path, 'static')
    img_path = os.path.join(static_path, 'banners/' + name)

    if os.path.exists(img_path):
        return send_from_directory(app_config['STATIC_ROOT'] + '/banners', name)

    return send_from_directory(app_config['STATIC_ROOT'], 'notfound.png')
