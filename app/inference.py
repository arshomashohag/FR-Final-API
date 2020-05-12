import keras
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from app.data.dataset import ApparelDataset
from app.config import app_config
from app.models.CustomModel import ImageEmbedding
import app.utils.utils as utils

# Input Shape
img_width, img_height, _ = 224, 224, 3  # load_image(df.iloc[0].image).shape

graph = tf.get_default_graph()

sess = tf.Session(graph=graph)
keras.backend.set_session(sess)

embedding_model = load_model(app_config['MODEL_WEIGHT_PATH'] + '/embedding-calculator.h5')
classifier_model = load_model(app_config['MODEL_WEIGHT_PATH'] + '/classifier.h5')


def classify_image(name_labels, other_labels, path):
    """
    Classify the given image into 15 sub-categories

    :param name_labels: labels of the 15 sub-categories
    :param other_labels: labels of all the other categories
    :param path: path of the query image
    :return: list of all predicted sub-classes
    """

    if not os.path.exists(path):
        print('Invalid image path')
        return

    keras.backend.clear_session()
    # Reshape
    img = load_img(path, target_size=(img_width, img_height))
    # img to Array
    x = img_to_array(img)
    # Expand Dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # Pre process Input
    x = preprocess_input(x)

    global sess
    global graph

    with graph.as_default():
        keras.backend.set_session(sess)
        x = classifier_model.predict(x).reshape(-1)
        threshold_flag = False

        # result = np.where(x == np.amax(x))
        max_ind = 0
        max_val = -1

        second_max_ind = 0
        second_max_val = -1

        ind = 0
        for prediction in x:
            if prediction > max_val:
                second_max_ind = max_ind
                second_max_val = max_val
                max_val = prediction
                max_ind = ind
            else:
                if prediction > second_max_val:
                    second_max_val = prediction
                    second_max_ind = ind

            if prediction >= app_config['CLASSIFIER_THRESHOLD']:
                threshold_flag = True
            ind = ind + 1

        print(x)
        if threshold_flag:
            return [name_labels[max_ind]], max_val
        else:
            other_labels.append(name_labels[max_ind])

            if second_max_val > 0.2:
                other_labels.append(name_labels[second_max_ind])
            print(other_labels)
            return other_labels, max_val


def get_embeddings(image_path):
    """
    Calculate image feature map or embedding of a given image

    :param image_path: path of the image
    :return: embedding of the image if successfully extracted the feature, empty array otherwise
    """
    keras.backend.clear_session()
    if os.path.exists(image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
        # img to Array
        x = keras.preprocessing.image.img_to_array(img)
        # Expand Dim (1, w, h)
        x = np.expand_dims(x, axis=0)
        # Pre process Input
        x = preprocess_input(x)

        global sess
        global graph

        with graph.as_default():
            keras.backend.set_session(sess)
            y = embedding_model.predict(x).reshape(-1).tolist()
            return y
    else:
        return []


def load_model_from_disk(model_name):
    """
    Load a model from the disk

    :param model_name: name of the model to be loaded
    :return: loaded model
    """
    try:
        model = load_model(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))
        return model
    except ImportError as err:
        print('Model not available, please train first')
        return err
    except IOError as err:
        print('Invalid file')
        return err


class Inference:
    """
        Model Inference
    """

    # Constructor to initialize and load data, embedding and recommendation table of items
    def __init__(self):
        """
        Initiate the inference instance
        """
        self.apparel_meta = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT'])
        self.embedding_model = ImageEmbedding()
        self.embedding_map = utils.load_from_pickle('embeddings')
        self.candidate_images_ids = utils.load_from_pickle('candidate_images')
        self.candidate_images = self.apparel_meta.filter_by_ids(self.candidate_images_ids)
        self.candidate_images['emb'] = self.candidate_images.apply(lambda row: self.embedding_map[row['id']], axis=1)
        self.recommendations = []

    # Recommend similar items to another item in the inventory by id
    def recommend_by_id(self, image_id):
        """
        Recommend product similar to another product in the inventory.

        :param image_id: id of the  image of the query product
        :return: top 10 most similar recommended product
        """
        filtered_meta = self.apparel_meta.filter_by_id(image_id)
        if filtered_meta.shape[0] <= 0:
            print('No image found')
            return

        filtered_meta['emb'] = filtered_meta.apply(lambda row: self.embedding_map[row['id']], axis=1)
        self.recommendations = utils.get_top_100_similar_product(self.embedding_map[image_id], filtered_meta)

        return self.recommendations

    # Recommend similar items to a query image
    def recommend_by_image(self, image_path, article_type=None, gender=None, color=None):
        """
        Recommend available products similar to an unknown query product

        :param color: color of the apparel
        :param gender: gender of the product to search for
        :param article_type: article_type of the product
        :param image_path: path of the query image
        :return: top 10 most similar products
        """
        all_metadata = self.apparel_meta.get_all_meta()

        try:
            new_img_embedding = get_embeddings(image_path)
            labels = ['Bags', 'Belts', 'Bottomwear', 'Eyewear', 'Flip Flops', 'Fragrance', 'Innerwear', 'Jewellery',
                      'Lips', 'Sandal', 'Shoes', 'Socks', 'Topwear', 'Wallets', 'Watches']
            other_labels = ["Dress", "Loungewear and Nightwear", "Saree", "Nails", "Makeup", "Headwear", "Ties",
                            "Accessories", "Scarves", "Cufflinks", "Apparel Set", "Free Gifts", "Stoles", "Skin Care",
                            "Skin", "Eyes", "Mufflers", "Shoe Accessories", "Sports Equipment", "Gloves", "Hair",
                            "Bath and Body", "Water Bottle", "Perfumes", "Umbrellas", "Wristbands",
                            "Beauty Accessories", "Sports Accessories", "Vouchers", "Home Furnishing"]

            temp_candidate_images = self.candidate_images

            if article_type is None:
                print('Classifying')
                sub_categories, sub_cat_sim_score = classify_image(labels, other_labels, image_path)
                print(sub_categories)
                modified_metadata = all_metadata[all_metadata['subCategory'].isin(sub_categories)]

            else:
                temp_candidate_images = temp_candidate_images[temp_candidate_images['articleType'] == article_type]
                modified_metadata = all_metadata[all_metadata['articleType'] == article_type]

            if gender is not None:
                modified_metadata = modified_metadata[modified_metadata['gender'] == gender]
                temp_candidate_images = temp_candidate_images[temp_candidate_images['gender'] == gender]

            if color is not None:
                modified_metadata = modified_metadata[modified_metadata['baseColour'] == color]
                temp_candidate_images = temp_candidate_images[temp_candidate_images['baseColour'] == color]

            if gender is None or article_type is None:
                gender, article_type, candidate_sim_score = utils.get_article_type(new_img_embedding,
                                                                                   temp_candidate_images)
                modified_candidate_metadata = []
                if gender is None:
                    modified_candidate_metadata = modified_metadata[modified_metadata['gender'] == gender]

                if article_type is None:
                    if gender is None:
                        modified_candidate_metadata = modified_candidate_metadata[
                            modified_candidate_metadata['articleType'] == article_type
                            ]
                    else:
                        modified_candidate_metadata = modified_metadata[
                            modified_metadata['articleType'] == article_type]

                if len(modified_candidate_metadata) > 0:
                    modified_metadata = modified_candidate_metadata
                else:
                    if sub_cat_sim_score < candidate_sim_score:
                        modified_metadata = all_metadata[all_metadata['articleType'] == article_type]
                        modified_metadata = modified_metadata[modified_metadata['gender'] == gender]

            modified_metadata['emb'] = modified_metadata.apply(lambda row: self.embedding_map[row['id']], axis=1)
            print(article_type)
            print(len(modified_metadata))
            self.recommendations = utils.get_top_100_similar_product(new_img_embedding, modified_metadata)

            # self.apparel_meta.filter_by_sub_categories(sub_categories)
            return self.recommendations

        except IOError as io_error:
            print('Invalid model file, train and save a valid model')
            return pd.DataFrame()
        except ImportError as import_error:
            print('No model found, train and save the model first')
            return pd.DataFrame()
        except Exception as exception:
            print('Unknown error')
            print(exception)
            return pd.DataFrame()
        finally:
            print('Done...')

    # SHow recommendations through plotting figure
    def show_recommendation(self):
        """
        Show the recommended products to the screen

        :return:
        """
        utils.plot_figures(self.recommendations, nrows=2, ncols=5)
        return


def print_ok():
    print('Ok')
