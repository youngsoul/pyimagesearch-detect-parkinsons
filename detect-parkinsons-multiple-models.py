
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os
from sklearn.externals import joblib
from xgboost import XGBClassifier
from skimage import feature
from imutils import paths
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from imutils import build_montages

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
    "knn": KNeighborsClassifier(n_neighbors=3),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf=4),
    # "mlp2": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=0.0001,
    #                       solver='adam', verbose=10, tol=0.000000001),
    # "mlp": MLPClassifier(),
    "xgboost": XGBClassifier(learning_rate=0.01)

}

"""
--dataset detect-parkinsons/dataset/spiral --model all
--dataset detect-parkinsons/dataset/wave --model all

"""


def get_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="animals",
                    help="path to directory containing the '3scenes' dataset")
    ap.add_argument("-m", "--model", type=str, default="all",
                    help="type of python machine learning model to use")
    args = vars(ap.parse_args())

    return args


def create_image_hog(image):
    """
    HOG is a structural descriptor that will capture and quantify changes in local gradient in teh input image
    HOG will be able to quantify how the directions of both sprials and waves change
    """
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image,
                           orientations=9,
                           pixels_per_cell=(10, 10),
                           cells_per_block=(2, 2),
                           transform_sqrt=True,
                           block_norm="L1"
                           )

    # resulting features are a 12,996-dim feature vector (list of numbers) quantifying the wave or spiral
    return features


def load_split(path):
    # grab the list of images in the input directory, then initialize the list of data
    # (i.e. images ) and class labels
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        # threshold the image such that the drawing appears as white on a black background
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # get hog features of image
        features = create_image_hog(image)

        # update the data and labels lists,
        data.append(features)
        labels.append(label)

    return np.array(data), np.array(labels), imagePaths


def get_image_features_and_labels(dataset_path):
    """

    :param dataset_path:
    :return:  features, labels. we are going to recombine training, testing and let crossval divide up
    """
    # define the path to the training and testing directories
    trainingPath = os.path.sep.join([dataset_path, "training"])
    testingPath = os.path.sep.join([dataset_path, "testing"])

    # load the training and testing data
    (X_train, y_train, trainImagePaths) = load_split(trainingPath)
    (X_test, y_test, testImagePaths) = load_split(testingPath)

    X = []
    y = []
    imagePaths = []
    X.extend(X_train)
    X.extend(X_test)
    y.extend(y_train)
    y.extend(y_test)
    imagePaths.extend(trainImagePaths)
    imagePaths.extend(testImagePaths)

    return X, y, imagePaths


def one_hot_encode_targets(target_values, dataset_name):
    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    numeric_labels = le.fit_transform(target_values)
    print(le.classes_)
    with open(f'./{dataset_name}_labels.txt', 'w') as f:
        for i, target_name in enumerate(le.classes_):
            f.write(f"{i},{target_name}")
            f.write("\n")

    return numeric_labels, le.classes_


def cross_validate_model(model_name, X, y):
    # train the model
    # print("[INFO] using '{}' model".format(args["model"]))
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=5)
    accuracy = scores.mean()

    return model_name, accuracy, model


def run_model_on_holdout(model, testX, testY, y_classes):
    metrics = {}

    # print(f'Model: \n {model}')
    # make predictions on our data and show a classification report
    # print("[INFO] evaluating...")
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)
    class_report = classification_report(testY, predictions,
                                         target_names=y_classes)

    # compute the confusion matrix and use it to derive the raw accuracy, sensitivity and specificity
    # Sensitivity measures the true positives that were also predicted as positives
    # Specificity measures the true negatives that were also predicted as negative
    cm = confusion_matrix(testY, predictions).flatten()
    (tn, fp, fn, tp) = cm
    # metrics['acc'] = (tp + tn) / float(cm.sum())
    metrics['acc_score'] = accuracy
    metrics['sensitivity'] = tp / float(tp + fn)
    metrics['specificity'] = tn / float(tn + fp)
    metrics['classification_report'] = class_report
    metrics['confusion_matrix'] = cm
    metrics['predictions'] = predictions

    return metrics


def display_predictions(model_name, holdout_images, y_classes, predictions, holdoutY ):
    output_images = []

    # loop over the testing/holdout samples
    for i, holdout_image in enumerate(holdout_images):
        # load the testing image, clone it, and resize it
        image = cv2.imread(holdout_image)
        output = image.copy()
        output = cv2.resize(output, (200,200))

        # draw the colored class label on the output image and add it to
        # the set of output images
        color = (0,255,0) if 'healthy' in holdout_image else (0,0,255)
        if holdoutY[i] != predictions[i]:
            if holdoutY[i] == 1:
                # then this means the actual was parkinsons, but we said healthy.
                # FN is worse than a FP.
                color = (172,31,211)
            else:
                color = (255,0,0)

        cv2.putText(output, f"{y_classes[holdoutY[i]]}/{y_classes[predictions[i]]}", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        output_images.append(output)

    # create a montage using 128x128 "tiles" with 5 rows and 5 columns
    montage = build_montages(output_images, (200, 200), (5, 5))[0]

    # show the output montage
    cv2.imshow(f"{model_name} actual/predicted", montage)
    cv2.waitKey(0)

def get_model_result_sort_key(x):
    return x[1]


if __name__ == '__main__':
    args = get_arguments()
    dataset_dir_name = args["dataset"]
    X, y, imagePaths = get_image_features_and_labels(dataset_dir_name)
    y_transformed, y_classes = one_hot_encode_targets(y, dataset_dir_name)

    X = pd.DataFrame(X)
    X['imagepath'] = imagePaths

    # perform a training and testing split, using 75% of the data for
    # training and 25% for evaluation
    (trainX, holdoutX, trainY, holdoutY) = train_test_split(X, y_transformed,
                                                            test_size=0.20)

    # save the image paths of the holdout set so we can show them
    holdout_images = holdoutX['imagepath']
    holdoutX.drop(columns=['imagepath'], inplace=True)
    trainX.drop(columns=['imagepath'], inplace=True)

    model_name = args["model"]

    results = []
    if model_name == 'all':
        for k, v in models.items():
            results.append(cross_validate_model(k, trainX, trainY))

        sorted_models = sorted(results, key=get_model_result_sort_key, reverse=True)
        for model_result in sorted_models:
            print("----------------------------------")
            print(model_result)
            print("----------------------------------")

        print("Best Model")
        print(sorted_models[0])
        best_model = sorted_models[0][2]
        best_model.fit(trainX, trainY)

        metrics = run_model_on_holdout(best_model, holdoutX, holdoutY, y_classes)
        print("Holdout set metrics")
        print(metrics)
        cm = metrics['confusion_matrix']
        df_cm = pd.DataFrame(
            cm.reshape(-1,2), index=['healthy', 'parkinson'], columns=['healthy', 'parkinson'],
        )
        print(df_cm.head())
        print(metrics['classification_report'])


        predictions = metrics['predictions']
        display_predictions(sorted_models[0][0], holdout_images, y_classes, predictions, holdoutY)


        saved_model_name = f"{dataset_dir_name}_image_classify_scikit_model.sav"
        with open(f'./{dataset_dir_name}_best_model_details.txt', 'w') as f:
            f.write(saved_model_name)
            f.write("\n")
            f.write(f"{sorted_models[0]}")

        print(f"Saving model to: {saved_model_name}")
        joblib.dump(best_model, saved_model_name)


    else:
        model_name, accuracy, model = cross_validate_model(model_name, trainX, trainY)
        print("Cross Validation Accuracy")
        print(model_name, accuracy)
        model.fit(trainX, trainY)

        print("--------------------------------------------------------------------")

        metrics = run_model_on_holdout(model, holdoutX, holdoutY, y_classes)
        print("Holdout Accuracy")
        print(metrics)
        cm = metrics['confusion_matrix']
        df_cm = pd.DataFrame(
            cm.reshape(-1,2), index=['healthy', 'parkinson'], columns=['healthy', 'parkinson'],
        )
        print(df_cm.head())
        print(metrics['classification_report'])

        predictions = metrics['predictions']

        saved_model_name = f"{dataset_dir_name}_image_classify_scikit_model.sav"
        joblib.dump(model, saved_model_name)
        print(f"Saving model to: {saved_model_name}")
        with open(f'./{dataset_dir_name}_{model_name}_model_details.txt', 'w') as f:
            f.write(f"{model}")
            f.write(f"\nAccuracy: {accuracy}")

        display_predictions(model_name, holdout_images, y_classes, predictions, holdoutY)
