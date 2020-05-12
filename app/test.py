import numpy as np
import pandas as pd
from app.config import app_config
import os
import app.utils.metric as metric


class ModelTesting:
    """Mode Testing"""

    def __init__(self, test_generator, model):
        """
        Initiate model testing instance

        :param test_generator: data generator for test data
        :param model: model to be tested
        """
        self.test_generator = test_generator
        self.model = model

    def test_model(self, class_indices, save_as_csv=True, output_file_name='prediction.csv'):
        """
        Test the model. Calculate accuracy and if permitted save the result of the prediction in a csv file

        :param class_indices: indices of the classes in the test set.
        :param save_as_csv: boolean flag whether to save predictions in csv, default True
        :param output_file_name: output file name, default prediction.csv
        :return: accuracy, class-wise right & wrong predictions count
        """
        filenames = self.test_generator.filenames
        nb_samples = len(filenames)
        predict = self.model.predict_generator(self.test_generator, steps=nb_samples, verbose=1)
        predicted_class_indices = np.argmax(predict, axis=1)

        labels = dict((v, k) for k, v in class_indices.items())
        predictions = [labels[k] for k in predicted_class_indices]

        results = pd.DataFrame({"Filename": filenames,
                                })

        for k, v in labels.items():
            results[v] = predict[:, k]

        results['Actual'] = results.apply(lambda row: row['Filename'].split('/')[0], axis=1)
        results["Predictions"] = predictions

        if save_as_csv:
            results.to_csv(os.path.join(app_config['MODEL_LOG_PATH'], output_file_name), index=False)

        return metric.calculate_accuracy(filenames, predictions)
