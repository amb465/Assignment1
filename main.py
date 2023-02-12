import os
import pandas as pd
import algorithms_tune
import algorithms_learn
import algorithms_train
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing


def generate_folder_structure():
    cwd = os.getcwd()
    print('Checking for folder structure')
    if not os.path.isdir(f'{cwd}/images'):
        os.mkdir(f'{cwd}/images')
        print('Created /images directory')
    if not os.path.isdir(f'{cwd}/images/tuning'):
        os.mkdir(f'{cwd}/images/tuning')
        print('Created /images/tuning directory')
    if not os.path.isdir(f'{cwd}/images/learning'):
        os.mkdir(f'{cwd}/images/learning')
        print('Created /images/learning directory')
    if not os.path.isdir(f'{cwd}/images/training'):
        os.mkdir(f'{cwd}/images/training')
        print('Created /images/training directory')

    if not os.path.isdir(f'{cwd}/notes'):
        os.mkdir(f'{cwd}/notes')
        print('Created /notes directory')
    if not os.path.isdir(f'{cwd}/notes/tuning'):
        os.mkdir(f'{cwd}/notes/tuning')
        print('Created /notes/tuning directory')
    if not os.path.isdir(f'{cwd}/notes/learning'):
        os.mkdir(f'{cwd}/notes/learning')
        print('Created /notes/learning directory')
    if not os.path.isdir(f'{cwd}/notes/training'):
        os.mkdir(f'{cwd}/notes/training')
        print('Created /notes/training directory')


def load_content(
    file_name,
    title,
    classification_column
):
    print(f'Process Input File {file_name} for {title}')
    data_set = pd.read_csv(file_name)
    x = data_set.drop(classification_column, axis=1)
    y = data_set[classification_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=training_split,
        stratify=y,
        random_state=random_seed)
    print('------------------------------------------------------------------')
    return x_train, x_test, y_train, y_test, x, y


if __name__ == "__main__":
    generate_folder_structure()
    training_split = 0.80
    random_seed = 1

    # High level testing drivers
    tuning = False
    learning = False
    training = True

    # For debugging, set to false to skip
    dt = False
    knn = False
    nn = False
    svm = True
    boost = False

    # File 1
    file_name_1 = 'beans.csv'
    title_1 = 'Bean Classification'
    classification_column_1 = 'Class'

    x_train_1, x_test_1, y_train_1, y_test_1, x_1, y_1 = load_content(
        file_name_1,
        title_1,
        classification_column_1)

    # File 2
    file_name_2 = 'phishing.csv'
    title_2 = 'Phishing'
    classification_column_2 = 'result'

    x_train_2, x_test_2, y_train_2, y_test_2, x_2, y_2 = load_content(
        file_name_2,
        title_2,
        classification_column_2)

    # Tuning

    if tuning:
        if dt:
            print(f'DT Processing for {title_1}')
            algorithms_tune.decision_tree_tune(
                x=x_train_1,
                y=y_train_1,
                max_levels=25,
                folds=5,
                title=title_1)

            print(f'DT Processing for {title_2}')
            algorithms_tune.decision_tree_tune(
                x=x_train_2,
                y=y_train_2,
                max_levels=25,
                folds=5,
                title=title_2)

        if nn:
            print(f'NN Processing for {title_1}')
            algorithms_tune.neural_network_tune(
                x=x_train_1,
                y=y_train_1,
                folds=5,
                title=title_1,
                scale=False)

            print(f'NN Processing for {title_1} - Scaled')
            algorithms_tune.neural_network_tune(
                x=x_train_1,
                y=y_train_1,
                folds=5,
                title=title_1,
                scale=True)

            print(f'NN Processing for {title_2}')
            algorithms_tune.neural_network_tune(
                x=x_train_2,
                y=y_train_2,
                folds=5,
                title=title_2,
                scale=False)

        if boost:
            print(f'Boosting Processing for {title_1}')
            algorithms_tune.boosting_tune(
                x=x_train_1,
                y=y_train_1,
                title=title_1,
                depth=9,
                min_samples=4,
                folds=5)
            print(f'Boosting Processing for {title_2}')
            algorithms_tune.boosting_tune(
                x=x_train_2,
                y=y_train_2,
                title=title_2,
                depth=6,
                min_samples=1,
                folds=5)

        if svm:
            print(f'SVM Processing for {title_1}')
            algorithms_tune.svm_tune(
                x=x_train_1,
                y=y_train_1,
                folds=5,
                title=title_1)

            print(f'SVM Processing for {title_2}')
            algorithms_tune.svm_tune(
                x=x_train_2,
                y=y_train_2,
                folds=5,
                title=title_2)

        if knn:
            print(f'KNN Processing for {title_1}')
            algorithms_tune.knn_tune(
                x=x_train_1,
                y=y_train_1,
                max_k=100,
                folds=5,
                title=title_1)

            print(f'KNN Processing for {title_2}')
            algorithms_tune.knn_tune(
                x=x_train_2,
                y=y_train_2,
                max_k=100,
                folds=5,
                title=title_2)

    if learning:
        if dt:
            print(f'Learning curve for DT for {title_1}')
            algorithms_learn.decision_tree_learn(
                x=x_1,
                y=y_1,
                max_depth=9,
                min_samples=4,
                title=title_1)

            print(f'Learning curve for DT for {title_2}')
            algorithms_learn.decision_tree_learn(
                x=x_2,
                y=y_2,
                max_depth=6,
                min_samples=1,
                title=title_2)

        if nn:
            print(f'Learning curve for NN for {title_1}')
            algorithms_learn.neural_network_learn(
                x=x_1,
                y=y_1,
                neurons=65,
                learning_rate=0.001,
                scale=True,
                title=title_1
            )

            print(f'Learning curve for NN for {title_2}')
            algorithms_learn.neural_network_learn(
                x=x_2,
                y=y_2,
                neurons=9,
                learning_rate=0.0001,
                scale=False,
                title=title_2
            )

        if boost:
            print(f'Learning curve for Boost for {title_1}')
            algorithms_learn.boosting_learn(
                x=x_1,
                y=y_1,
                depth=2,
                min_samples=4,
                learning_rate=0.1,
                title=title_1
            )

            print('Learning curve for Boost for Bean-NP')
            algorithms_learn.boosting_learn(
                x=x_1,
                y=y_1,
                depth=9,
                min_samples=4,
                learning_rate=1,
                title='Bean-NP'
            )

            print(f'Learning curve for Boost for {title_2}')
            algorithms_learn.boosting_learn(
                x=x_2,
                y=y_2,
                depth=2,
                min_samples=1,
                learning_rate=0.1,
                title=title_2
            )

        if svm:
            print(f'Learning curve for SVM for {title_1}')
            algorithms_learn.svm_learn(
                x=x_1,
                y=y_1,
                kernel='linear',
                C=2,
                shrink=False,
                degree=0,
                title=title_1
            )
            print(f'Learning curve for SVM for {title_2}')
            algorithms_learn.svm_learn(
                x=x_2,
                y=y_2,
                kernel='poly',
                degree=10,
                C=2,
                shrink=True,
                title=title_2
            )

        if knn:
            print(f'Learning curve for KNN for {title_1}')
            algorithms_learn.knn_learn(
                x=x_1,
                y=y_1,
                k=5,
                distance=1,
                weights='distance',
                title=title_1)

            print(f'Learning curve for KNN for {title_2}')
            algorithms_learn.knn_learn(
                x=x_2,
                y=y_2,
                k=12,
                distance=1,
                weights='distance',
                title=title_2)

    if training:
        if dt:
            algorithms_train.decision_tree_train(
                x_train=x_train_1,
                x_test=x_test_1,
                y_train=y_train_1,
                y_test=y_test_1,
                depth=9,
                min_samples=4,
                title=title_1
            )
            algorithms_train.decision_tree_train(
                x_train=x_train_2,
                x_test=x_test_2,
                y_train=y_train_2,
                y_test=y_test_2,
                depth=6,
                min_samples=1,
                title=title_2
            )

        if nn:
            algorithms_train.neural_network_train(
                x_train=x_train_1,
                x_test=x_test_1,
                y_train=y_train_1,
                y_test=y_test_1,
                neurons=65,
                learning_rate=0.001,
                scale=True,
                title=title_1
            )
            algorithms_train.neural_network_train(
                x_train=x_train_2,
                x_test=x_test_2,
                y_train=y_train_2,
                y_test=y_test_2,
                neurons=9,
                learning_rate=0.0001,
                scale=False,
                title=title_2
            )

        if boost:
            algorithms_train.boosting_train(
                x_train=x_train_1,
                x_test=x_test_1,
                y_train=y_train_1,
                y_test=y_test_1,
                depth=2,
                min_samples=4,
                learning_rate=0.1,
                title=title_1
            )
            algorithms_train.boosting_train(
                x_train=x_train_2,
                x_test=x_test_2,
                y_train=y_train_2,
                y_test=y_test_2,
                depth=2,
                min_samples=1,
                learning_rate=0.1,
                title=title_2
            )

        if svm:
            algorithms_train.svm_train(
                x_train=x_train_1,
                x_test=x_test_1,
                y_train=y_train_1,
                y_test=y_test_1,
                kernel='linear',
                degree=0,
                C=2,
                shrink=False,
                title=title_1
            )

            algorithms_train.svm_train(
                x_train=x_train_2,
                x_test=x_test_2,
                y_train=y_train_2,
                y_test=y_test_2,
                kernel='poly',
                degree=10,
                C=2,
                shrink=True,
                title=title_2
            )

        if knn:
            algorithms_train.knn_train(
                x_train=x_train_1,
                x_test=x_test_1,
                y_train=y_train_1,
                y_test=y_test_1,
                k=5,
                distance=1,
                weights='distance',
                title=title_1
            )

            algorithms_train.knn_train(
                x_train=x_train_2,
                x_test=x_test_2,
                y_train=y_train_2,
                y_test=y_test_2,
                k=12,
                distance=1,
                weights='distance',
                title=title_2
            )
