import os
from app.config import app_config
import shutil
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import pairwise_distances
from scipy import spatial
import pickle


def plot_figures(figures, nrows=1, ncols=1, figsize=(5, 5)):
    """Plot a list of figures(images) in grid system

    Parameters
    ----------
    :param figures : (id, similarity_score, article_type) list of tuple
    :param ncols : number of columns of subplots wanted in the display
    :param nrows : number of rows of subplots wanted in the figure
    :param figsize: size of the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind in range(len(figures)):
        fig = figures[ind]
        img = read_image_by_id(fig[0])
        axeslist.ravel()[ind].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(str(fig[0]) + '-' + fig[2])
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    return


def read_image_by_id(image_id):
    """
    Read an image from inventory by it's id.

    Parameters
    ----------
    :param image_id: id(1528) of the image
    :return: 224X224 resized version of the loaded image(1528.jpg)
    """
    path = os.path.join(app_config['DATA_IMAGE_ROOT'], str(image_id) + '.jpg')
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.resize(img, (224, 224))
    print('No image found!')


def get_article_type(emb_new_item, candidate_image_meta, sub_class_article_gender=None):
    """
    Calculate similarity between a new and unknown image and all the candidate product images.
    Return the gender and article type of most similar candidate

    Parameters
    ----------
    :param sub_class_article_gender: Concatenated gender and article of products in classified sub-categories
    :param emb_new_item: feature map or embedding of new image
    :param candidate_image_meta: all candidate image meta with their respective embeddings
    :return: gender and articleType of the most similar candidate product image
    """
    if candidate_image_meta is None:
        print('No candidate image found!')
        return

    if len(candidate_image_meta) <= 0:
        print('No candidate found')
        return

    tmp = 0
    most_similar = None

    most_similar_in_subcategory = None
    temp_msis = 0
    unique_gender_article = None
    if sub_class_article_gender is not None:
        unique_gender_article = sub_class_article_gender.unique()

    for ind, row in candidate_image_meta.iterrows():
        sim_score = 1 - spatial.distance.cosine(emb_new_item, row.emb)
        if sim_score > tmp:
            most_similar = row
            tmp = sim_score
        if unique_gender_article is not None and row.genderArticle in unique_gender_article:
            if sim_score > temp_msis:
                most_similar_in_subcategory = row
                temp_msis = sim_score
    print(tmp)
    if most_similar_in_subcategory is not None:
        return most_similar['gender'], most_similar['articleType'], tmp, most_similar_in_subcategory['articleType'], \
               temp_msis
    return most_similar['gender'], most_similar['articleType'], tmp, None, None


def generate_candidates(apparel_data, save_as_pickle=True):
    """
    Generate candidate images from all the inventory apparel images. Separate the images into several groups by
    concatenating  gender and article type and then pick a candidate from each group which is most similar to all
    the images to that group.

    Parameters
    ----------
    :param apparel_data: metadata of all the apparel in the inventory
    :param save_as_pickle: boolean flag whether user wants to save the candidate images id in a pickle file
    :return: True if successful, raise error otherwise
    """
    candidates = []
    unique_types = set(apparel_data['genderArticle'])
    print('Unique types ' + str(len(unique_types)))
    for group in unique_types:
        apparel_data_group = apparel_data[apparel_data['genderArticle'] == group]
        print('Group ' + group)
        print(len(apparel_data_group))
        score_map = {}
        embeddings_group = []
        for ind, row in apparel_data_group.iterrows():
            embeddings_group.append(row.emb)

        sim_scores = 1 - pairwise_distances(embeddings_group, metric='cosine')

        i = 0
        for inner_ind, data in apparel_data_group.iterrows():
            score_map[data.id] = sum(sim_scores[i])
            i += 1

        candidates.append(max(score_map, key=score_map.get))

    print(len(candidates))
    if save_as_pickle:
        save_to_pickle(candidates, 'candidate_images')

    return True


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file named file_name

    Parameters
    ----------
    :param data: data to save in the file
    :param file_name: name of the file
    :return: True if successful False otherwise
    """
    try:
        candidate_meta_file = open(os.path.join(app_config['PROCESSED_FILE_PATH'], file_name), 'wb')
        pickle.dump(data, candidate_meta_file)
        candidate_meta_file.close()
        print('Saved...')
        return True
    except Exception as error:
        print('Error occurred while saving in file')
        print(error)
        return False
    finally:
        print('Done...')


def load_from_pickle(file_name):
    """
    Load a pickle file from data/processed file directory

    Parameters
    ----------
    :param file_name: name of the file to be loaded
    :return: data in the file if successful, empty array otherwise
    """
    try:
        file = open(os.path.join(app_config['PROCESSED_FILE_PATH'], file_name), 'rb')
        return pickle.load(file)
    except Exception as error:
        print('Error reading pickle file')
        print(error)
        return []
    finally:
        print('Done...')


def get_top_100_similar_product(query_image_embedding, filtered_meta):
    """
    Calculate similarity between a query product image and some other product images from inventory.
    Return top 10 most similar of them

    Parameters
    ----------
    :param query_image_embedding: feature map or embedding of the query image
    :param filtered_meta: dataframe containing some other product details
    :return: list of tuples containing data of top 10 most similar products in the inventory
    """
    filtered_image_embeddings = []
    for ind, meta in filtered_meta.iterrows():
        if len(meta.emb) != 2048:
            print(meta.id)
            meta.emb = [0] * 2048
        filtered_image_embeddings.append(meta.emb)

    query_image_embedding = [query_image_embedding]

    sim_scores = 1 - pairwise_distances(query_image_embedding, filtered_image_embeddings, metric='cosine')
    sim_scores = sim_scores[0]

    recommendations = []
    i = 0
    for meta in filtered_meta.iterrows():
        recommendations.append(
            (meta[1]['id'],
             sim_scores[i],
             meta[1]['articleType'],
             meta[1]['productDisplayName'],
             meta[1]['gender'],
             meta[1]['baseColour']))
        i = i + 1

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommendations = recommendations[: 100]
    return recommendations


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app_config['ALLOWED_EXTENSIONS']


class InvalidValueError(Exception):
    """
    Custom exception for invalid parameter
    """

    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class ImageFileUtils:
    """
    Some utility for image files
    """

    def __init__(self, meta, column_name):
        self.meta = meta
        self.column_name = column_name

    # move all train/test/validation images to train/test/validation folder
    # train/test/validation folder contains same number of sub-folder for each class(column_name)
    def move_image_to_dir(self, dir_name):
        """
        Move the image files to dir_name/<category> directory.

        Parameters
        ----------
        :param dir_name: parent directory name such as train, test, validation
        :return: a dictionary containing number of images found and actually moved
        """
        actual_moved = {}
        image_path = app_config['DATA_IMAGE_ROOT']
        for i, img in self.meta.iterrows():

            if not os.path.exists(os.path.join(image_path, dir_name + '/' + img[self.column_name])):
                os.makedirs(os.path.join(image_path, dir_name + '/' + img[self.column_name]))

            if os.path.isfile(os.path.join(image_path, img['image'])):
                os.rename(os.path.join(image_path, img['image']),
                          os.path.join(image_path, dir_name + '/' + img[self.column_name] + '/' + img['image']))

                if img[self.column_name] in actual_moved:
                    actual_moved[img[self.column_name]] = actual_moved[img[self.column_name]] + 1
                else:
                    actual_moved[img[self.column_name]] = 1
        return actual_moved

    # move train/test/validation images in single directory(inventory)
    def move_images_to_inventory(self, dir_name):
        """
        Move the train/test/validation image back to inventory(root directory)

        Parameters
        ----------
        :param dir_name: name of the directory(train/test/validation) from where the image should move
        :return: boolean flag True for success, raise error otherwise
        """
        image_path = app_config['DATA_IMAGE_ROOT']

        for column_name in self.meta[self.column_name].unique():
            source = os.path.join(image_path, dir_name + '/' + column_name)

            if os.path.isdir(source):
                files = os.listdir(source)
                for f in files:
                    shutil.move(source + '/' + f, image_path)
                os.rmdir(source)
        return True
