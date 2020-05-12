from app import app
from app.config import app_config
from flask import request, jsonify, send_from_directory
from flask_cors import cross_origin
from werkzeug.utils import secure_filename

import app.utils.utils as utils
import time
import os

from app.inference import Inference
from app.data.dataset import ApparelDataset

inferer = Inference()
all_meta = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT']).get_all_meta()
all_meta.fillna('', inplace=True)


@app.route('/')
@cross_origin()
def index():
    return "Hello World"


@app.route('/search-by-id', methods=['POST'])
@cross_origin()
def search_by_id():
    # check if the post request has the file part
    if 'image_id' not in request.form:
        return jsonify({'success': False, 'message': 'Provide an image id'})
    try:
        image_id = int(request.form['image_id'])

        query_product = all_meta[all_meta['id'] == image_id]
        query_product_data = {}
        for ind, row in query_product.iterrows():
            for key, val in row.items():
                query_product_data[key] = val

        recommendations = inferer.recommend_by_id(image_id)

        data = []
        for rec in recommendations:
            data.append({
                'id': str(rec[0]),
                'similarity': str(rec[1]),
                'articleType': rec[2],
                'productDisplayName': rec[3]
            })

        return jsonify({
            'success': True,
            'query_image': query_product_data,
            'meta': [{}],
            'data': data
        })

    except Exception as error:
        print(error)
        return jsonify({
            'success': False,
            'message': 'Exception occurred'
        })
    finally:
        print('Done...')


@app.route('/recommend-by-image', methods=['POST'])
@cross_origin()
def search_by_image():
    if request.method == 'POST':
        start_time = time.time()

        static_path = os.path.join(app.root_path, 'static')

        if 'file' in request.files:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({'success': False, 'message': 'Invalid image'})
            if file and utils.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = static_path + '/query_images'
                file.save(os.path.join(filepath, filename))

                article_type = None
                gender = None
                color = None
                if 'article_type' in request.form:
                    article_type = request.form['article_type']

                if 'base_colour' in request.form:
                    color = request.form['base_colour']

                if 'gender' in request.form:
                    gender = request.form['gender']

                recommendations = inferer.recommend_by_image(os.path.join(app_config['QUERY_IMAGE_PATH'], filename),
                                                             article_type,
                                                             gender,
                                                             color)
                data = []
                for rec in recommendations:
                    data.append({
                        'id': str(rec[0]),
                        'similarity': str(rec[1]),
                        'articleType': rec[2],
                        'productDisplayName': rec[3]
                    })

                return jsonify({
                    'success': True,
                    'data': data
                })
            else:
                return jsonify({'success': False, 'message': 'Invalid image'})
        else:
            filtered_products = all_meta

            if 'master_category' in request.form:
                filtered_products = filtered_products[
                    filtered_products['masterCategory'] == request.form['master_category']
                ]

            if 'sub_category' in request.form:
                filtered_products = filtered_products[
                    filtered_products['subCategory'] == request.form['sub_category']
                    ]
            if 'article_type' in request.form:
                filtered_products = filtered_products[filtered_products['articleType'] == request.form['article_type']]

            if 'base_colour' in request.form:
                filtered_products = filtered_products[filtered_products['baseColour'] == request.form['base_colour']]

            if 'gender' in request.form:
                filtered_products = filtered_products[filtered_products['gender'] == request.form['gender']]

            response = []
            for ind, row in filtered_products.head(100).iterrows():
                data = {}
                for key, val in row.items():
                    data[key] = val
                response.append(data)
            return jsonify({'success': True, 'data': response})


@app.route('/home')
@cross_origin()
def get_home_values():
    # Execute query

    home_values = {
        'gender': all_meta.gender.unique().tolist(),
        'master_category': all_meta.masterCategory.unique().tolist(),
        'sub_category': all_meta.subCategory.unique().tolist(),
        'article_type': all_meta.articleType.unique().tolist(),
        'base_colour': all_meta.baseColour.unique().tolist(),
        'usage_type': all_meta.usage.unique().tolist(),
        'trending': [],
        'gender_wise_article': {'kids': [], 'men': [], 'women': []},
        'images': []
    }
    top_108 = all_meta.head(108)

    for ind, row in top_108.iterrows():
        data = {}
        for key, val in row.items():
            data[key] = val
        home_values['images'].append(data)

    trending_items = all_meta.groupby(['masterCategory']).head(8)
    gender_wise_article = all_meta.groupby(['gender']).head(4)

    for ind, row in trending_items.iterrows():
        data = {}
        for key, val in row.items():
            data[key] = val
        home_values['trending'].append(data)

    for ind, row in gender_wise_article.iterrows():

        if row['gender'] == 'Girls' or row['gender'] == 'Boys':
            home_values['gender_wise_article']['kids'].append(row['articleType'])
        elif row['gender'] == 'Men':
            home_values['gender_wise_article']['men'].append(row['articleType'])
        elif row['gender'] == 'Women':
            home_values['gender_wise_article']['women'].append(row['articleType'])

    return jsonify({
        'success': True,
        'data': home_values
    })
