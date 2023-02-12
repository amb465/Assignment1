from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from os import getcwd
import matplotlib.pyplot as plt


random_seed = 1
training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def log_out(display_screen, file_handler, message):
    if display_screen:
        print(message)
    file_handler.write(message+'\n')


def decision_tree_learn(x, y, max_depth, min_samples, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/learning/{title}-DT-learn.txt', 'w')

    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples,
        random_state=random_seed
    )
    train_size_abs, train_scores, test_scores = learning_curve(
        estimator=classifier,
        X=x,
        y=y,
        train_sizes=training_sizes
    )

    training_scores = []
    testing_scores = []
    train_sizes = []

    for train_size, cv_train_scores, cv_test_scores in zip(
         train_size_abs, train_scores, test_scores):
        train_sizes.append(train_size)

        train_score = cv_train_scores.mean()
        training_scores.append(train_score)

        test_score = cv_test_scores.mean()
        testing_scores.append(test_score)

        log_out(
            display_screen=True,
            file_handler=file,
            message=f'{train_size} samples were used to train the model'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average train accuracy is '
            f'{train_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average test accuracy is '
            f'{test_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message='\n'
        )

    fig, ax = plt.subplots()
    plt.title(f'DT Learning Curve for {title}')
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    ax.plot(train_sizes, training_scores, label='Training', color='blue')
    ax.plot(train_sizes, testing_scores, label='Testing', color='red')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/learning/{title}-DT-Learn.png')
    plt.close()
    file.close()


def neural_network_learn(x, y, neurons, learning_rate, title, scale):
    cwd = getcwd()
    file = open(f'{cwd}/notes/learning/{title}-NN-learn.txt', 'w')

    if scale:
        scaler = StandardScaler()
        scaler.fit(x)
        x_val = scaler.transform(x)
    else:
        x_val = x

    classifier = MLPClassifier(
        hidden_layer_sizes=neurons,
        max_iter=5000,
        learning_rate_init=learning_rate,
        random_state=random_seed)

    train_size_abs, train_scores, test_scores = learning_curve(
        estimator=classifier,
        X=x_val,
        y=y,
        train_sizes=training_sizes
    )

    training_scores = []
    testing_scores = []
    train_sizes = []

    for train_size, cv_train_scores, cv_test_scores in zip(
         train_size_abs, train_scores, test_scores):
        train_sizes.append(train_size)

        train_score = cv_train_scores.mean()
        training_scores.append(train_score)

        test_score = cv_test_scores.mean()
        testing_scores.append(test_score)

        log_out(
            display_screen=True,
            file_handler=file,
            message=f'{train_size} samples were used to train the model'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average train accuracy is '
            f'{train_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average test accuracy is '
            f'{test_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message='\n'
        )

    fig, ax = plt.subplots()
    plt.title(f'NN Learning Curve for {title}')
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    ax.plot(train_sizes, training_scores, label='Training', color='blue')
    ax.plot(train_sizes, testing_scores, label='Testing', color='red')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/learning/{title}-NN-Learn.png')
    plt.close()
    file.close()


def boosting_learn(x, y, depth, min_samples, learning_rate, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/learning/{title}-Boost-learn.txt', 'w')

    estimator = 2500

    tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=min_samples,
        random_state=random_seed
    )
    classifier = AdaBoostClassifier(
        n_estimators=estimator,
        estimator=tree,
        learning_rate=learning_rate)

    train_size_abs, train_scores, test_scores = learning_curve(
        estimator=classifier,
        X=x,
        y=y,
        train_sizes=training_sizes
    )

    training_scores = []
    testing_scores = []
    train_sizes = []

    for train_size, cv_train_scores, cv_test_scores in zip(
         train_size_abs, train_scores, test_scores):
        train_sizes.append(train_size)

        train_score = cv_train_scores.mean()
        training_scores.append(train_score)

        test_score = cv_test_scores.mean()
        testing_scores.append(test_score)

        log_out(
            display_screen=True,
            file_handler=file,
            message=f'{train_size} samples were used to train the model'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average train accuracy is '
            f'{train_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average test accuracy is '
            f'{test_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message='\n'
        )

    fig, ax = plt.subplots()
    plt.title(f'Boosting Learning Curve for {title}')
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    ax.plot(train_sizes, training_scores, label='Training', color='blue')
    ax.plot(train_sizes, testing_scores, label='Testing', color='red')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/learning/{title}-Boost-Learn.png')
    plt.close()
    file.close()


def svm_learn(x, y, kernel, degree, C, shrink, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/learning/{title}-svm-learn.txt', 'w')
    if degree != 0:
        classifier = SVC(
            kernel=kernel,
            degree=degree,
            C=C,
            shrinking=shrink,
            random_state=random_seed)
    else:
        classifier = SVC(
            kernel=kernel,
            C=C,
            shrinking=shrink,
            random_state=random_seed
        )
    train_size_abs, train_scores, test_scores = learning_curve(
        estimator=classifier,
        X=x,
        y=y,
        train_sizes=training_sizes
    )

    training_scores = []
    testing_scores = []
    train_sizes = []

    for train_size, cv_train_scores, cv_test_scores in zip(
         train_size_abs, train_scores, test_scores):
        train_sizes.append(train_size)

        train_score = cv_train_scores.mean()
        training_scores.append(train_score)

        test_score = cv_test_scores.mean()
        testing_scores.append(test_score)

        log_out(
            display_screen=True,
            file_handler=file,
            message=f'{train_size} samples were used to train the model'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average train accuracy is '
            f'{train_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average test accuracy is '
            f'{test_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message='\n'
        )

    fig, ax = plt.subplots()
    plt.title(f'SVM Learning Curve for {title}')
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    ax.plot(train_sizes, training_scores, label='Training', color='blue')
    ax.plot(train_sizes, testing_scores, label='Testing', color='red')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/learning/{title}-SVM-Learn.png')
    plt.close()
    file.close()


def knn_learn(x, y, k, distance, weights, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/learning/{title}-knn-learn.txt', 'w')

    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        p=distance)
    train_size_abs, train_scores, test_scores = learning_curve(
        estimator=classifier,
        X=x,
        y=y,
        train_sizes=training_sizes
    )

    training_scores = []
    testing_scores = []
    train_sizes = []

    for train_size, cv_train_scores, cv_test_scores in zip(
         train_size_abs, train_scores, test_scores):
        train_sizes.append(train_size)

        train_score = cv_train_scores.mean()
        training_scores.append(train_score)

        test_score = cv_test_scores.mean()
        testing_scores.append(test_score)

        log_out(
            display_screen=True,
            file_handler=file,
            message=f'{train_size} samples were used to train the model'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average train accuracy is '
            f'{train_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message=f'The average test accuracy is '
            f'{test_score:.2f}'
        )
        log_out(
            display_screen=True,
            file_handler=file,
            message='\n'
        )

    fig, ax = plt.subplots()
    plt.title(f'KNN Learning Curve for {title}')
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    ax.plot(train_sizes, training_scores, label='Training', color='blue')
    ax.plot(train_sizes, testing_scores, label='Testing', color='red')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/learning/{title}-KNN-Learn.png')
    plt.close()
    file.close()
