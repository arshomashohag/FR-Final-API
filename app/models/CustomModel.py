from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
from app.config import app_config
import os
import numpy as np

from app.data.dataset import ApparelDataset
import app.utils.utils as utils


class Classifier:
    """Image classifier"""

    def __init__(self, number_of_classes=15, learning_rate=0.00005, activation="softmax",
                 loss="categorical_crossentropy"):
        """
        Initiate an image classifier

        :param number_of_classes: number of classes of the model
        :param learning_rate: learning rate of the classifier model
        :param activation: activation function of the prediction layer
        :param loss: loss function of the classifier model
        """
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss

        self.base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        for layer in self.base_model.layers[:170]:
            layer.trainable = False

        x = self.base_model.output
        x = Flatten()(x)

        self.prediction = Dense(self.number_of_classes, activation=self.activation)(x)

        self.model = Model(inputs=self.base_model.input, outputs=self.prediction)

        self.model.compile(loss=self.loss,
                           optimizer=optimizers.SGD(lr=self.learning_rate),
                           metrics=['accuracy'])

    def get_model(self):
        """
        Get the classifier model

        :return: compiled classifier
        """
        return self.model

    def save_model_to_disk(self, model_name):
        """
        Save model to disk

        :param model_name: name of the model will be
        :return:
        """
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))


class ImageEmbedding:
    """Image Embedding"""

    def __init__(self):
        """
        Initiate the embedding class
        """
        self.input_shape = (224, 224, 3)
        self.base_model = ResNet50(weights='imagenet',
                                   include_top=False,
                                   input_shape=self.input_shape)
        self.base_model.trainable = False

        self.model = Sequential([
            self.base_model,
            GlobalMaxPooling2D()
        ])

    def get_model(self):
        """
         Get the embedding model

        :return: compiled feature extractor model
        """
        return self.model

    def save_model_to_disk(self, model_name):
        """
        Save model to disk

        :param model_name: name of the model will be
        :return:
        """
        self.model.save(os.path.join(app_config['MODEL_WEIGHT_PATH'], model_name))

    def get_embedding(self, img_name):
        """
        Extract feature from the image.

        :param img_name: name of the image
        :return: feature map of the image
        """
        image_path = os.path.join(app_config['DATA_IMAGE_ROOT'], img_name)
        # Reshape
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=(self.input_shape[0], self.input_shape[1]))
            # img to Array
            x = img_to_array(img)
            # Expand Dim (1, w, h)
            x = np.expand_dims(x, axis=0)
            # Pre process Input
            x = preprocess_input(x)
            return self.model.predict(x).reshape(-1)
        else:
            return [0] * self.model.output_shape[1]


class EmbeddingCalculator:

    def __init__(self):
        self.apparel_meta = ApparelDataset(app_config['DATA_LABEL_PATH'], app_config['DATA_IMAGE_ROOT'])
        self.embedding_model = ImageEmbedding()
        self.embedding_map = {}
        return

    # Calculate embedding of all items
    def calculate_all_embeddings(self):
        """
            Calculate embeddings of all the images in the inventory

            :return:
        """
        self.embedding_map = {}
        all_metadata = self.apparel_meta.get_all_meta()
        for meta in all_metadata.iterrows():
            if meta[1]['id'] not in self.embedding_map:
                self.embedding_map[meta[1]['id']] = self.embedding_model.get_embedding(meta[1]['image'])

    def save_embeddings_to_pickle(self):
        """
        Save calculated embeddings in a pickle file

        :return:
        """
        utils.save_to_pickle(self.embedding_map, 'embeddings')

    def generate_candidate_products(self):
        """
        Generate candidate products from different gender-article group

        :return: True if successful, raise error otherwise
        """
        all_meta = self.apparel_meta.get_all_meta()
        all_meta['emb'] = all_meta.apply(lambda row: self.embedding_map[row['id']], axis=1)
        utils.generate_candidates(all_meta, save_as_pickle=True)
        return True
