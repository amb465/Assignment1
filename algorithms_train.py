from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from os import getcwd
import time


random_seed = 1


def log_out(display_screen, file_handler, message):
    if display_screen:
        print(message)
    file_handler.write(message+'\n')


def decision_tree_train(
    x_train,
    x_test,
    y_train,
    y_test,
    depth,
    min_samples,
        title):

    cwd = getcwd()
    file = open(f'{cwd}/notes/training/{title}-DT-notes.txt', 'w')

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training DT for {title}')

    start = time.perf_counter()
    classifier = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=min_samples,
        random_state=random_seed)
    classifier.fit(x_train, y_train)

    y_train_prediction = classifier.predict(x_train)
    y_test_prediction = classifier.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    node_count = classifier.tree_.node_count
    end = time.perf_counter()

    training_time = end - start

    tree.plot_tree(classifier)
    plt.savefig(f'./images/training/{title}-pruned-tree.png')

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Node Count  : {node_count}'
    )

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Time  : {training_time}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction)}'
    )

    unprunned_classifier = DecisionTreeClassifier(
        random_state=random_seed)
    unprunned_classifier.fit(x_train, y_train)

    y_train_prediction_un = classifier.predict(x_train)
    y_test_prediction_un = classifier.predict(x_test)

    train_accuracy_un = accuracy_score(y_train, y_train_prediction_un)
    test_accuracy_un = accuracy_score(y_test, y_test_prediction_un)
    node_count_un = unprunned_classifier.tree_.node_count

    log_out(
        display_screen=True,
        file_handler=file,
        message='Unrpuned Tree'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Node Count  : {node_count_un}'
    )

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy_un}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy_un}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction_un)}'
    )
    tree.plot_tree(unprunned_classifier)
    plt.savefig(f'./images/training/{title}-unpruned-tree.png')
    plt.close()


def neural_network_train(
    x_train,
    x_test,
    y_train,
    y_test,
    neurons,
    learning_rate,
    scale,
        title):

    cwd = getcwd()
    file = open(f'{cwd}/notes/training/{title}-NN-notes.txt', 'w')

    if scale:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_val = scaler.transform(x_train)
        x_test_val = scaler.transform(x_test)
    else:
        x_train_val = x_train
        x_test_val = x_test

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training NN for {title}')

    start = time.perf_counter()
    classifier = MLPClassifier(
        hidden_layer_sizes=neurons,
        max_iter=5000,
        learning_rate_init=learning_rate,
        random_state=random_seed)
    classifier.fit(x_train_val, y_train)

    y_train_prediction = classifier.predict(x_train_val)
    y_test_prediction = classifier.predict(x_test_val)

    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    end = time.perf_counter()

    training_time = end - start

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Time  : {training_time}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction)}'
    )


def boosting_train(
    x_train,
    x_test,
    y_train,
    y_test,
    depth,
    min_samples,
    learning_rate,
        title):

    cwd = getcwd()
    file = open(f'{cwd}/notes/training/{title}-BOOST-notes.txt', 'w')

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training BOOST for {title}')

    start = time.perf_counter()

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
    classifier.fit(x_train, y_train)

    y_train_prediction = classifier.predict(x_train)
    y_test_prediction = classifier.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    end = time.perf_counter()

    training_time = end - start

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Time  : {training_time}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction)}'
    )


def svm_train(
    x_train,
    x_test,
    y_train,
    y_test,
    kernel,
    degree,
    C,
    shrink,
        title):

    cwd = getcwd()
    file = open(f'{cwd}/notes/training/{title}-SVM-notes.txt', 'w')

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training SVM for {title}')

    start = time.perf_counter()
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
            random_state=random_seed
        )
    classifier.fit(x_train, y_train)

    y_train_prediction = classifier.predict(x_train)
    y_test_prediction = classifier.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    end = time.perf_counter()

    training_time = end - start

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Time  : {training_time}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction)}'
    )


def knn_train(
    x_train,
    x_test,
    y_train,
    y_test,
    k,
    distance,
    weights,
        title):

    cwd = getcwd()
    file = open(f'{cwd}/notes/training/{title}-KNN-notes.txt', 'w')

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training KNN for {title}')

    start = time.perf_counter()
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        p=distance)
    classifier.fit(x_train, y_train)

    y_train_prediction = classifier.predict(x_train)
    y_test_prediction = classifier.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    end = time.perf_counter()

    training_time = end - start

    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Training Accuracy  : {train_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Testing Accuracy  : {test_accuracy}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Time  : {training_time}'
    )
    log_out(
        display_screen=True,
        file_handler=file,
        message=f'Confusion Matrix  : \n'
        f'{confusion_matrix(y_test, y_test_prediction)}'
    )
