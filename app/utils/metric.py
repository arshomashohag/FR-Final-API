

def calculate_accuracy(actual, prediction):
    """
    Calculate accuracy given actual and prediction

    :param actual: actual class names
    :param prediction: prediction of the model
    :return: accuracy, dictionary of class-wise right & wrong count
    """
    total = len(actual)
    correct = 0
    class_wise = {}
    for i, name in enumerate(actual):
        actual_name = name.split('/')[0]
        if prediction[i] == actual_name:

            if actual_name in class_wise:
                class_wise[actual_name]['right'] = class_wise[actual_name]['right'] + 1
            else:
                class_wise[actual_name] = {}
                class_wise[actual_name]['right'] = 1
                class_wise[actual_name]['wrong'] = 0
            correct = correct + 1
        else:
            if actual_name in class_wise:
                class_wise[actual_name]['wrong'] = class_wise[actual_name]['wrong'] + 1
            else:
                class_wise[actual_name] = {}
                class_wise[actual_name]['wrong'] = 1
                class_wise[actual_name]['right'] = 0

    return correct / total, class_wise
