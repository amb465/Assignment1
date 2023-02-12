from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
import time


plt.switch_backend('agg')
random_seed = 1


def log_out(display_screen, file_handler, message):
    if display_screen:
        print(message)
    file_handler.write(message+'\n')


def svm_tune(x, y, folds, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/tuning/{title}-SVM-tune.txt', 'w')

    c_values = [0.1, 0.5, 1, 1.5, 2]

    linear_scores_shrink = []
    linear_scores_no_shrink = []
    poly_3_score_shrink = []
    poly_3_score_no_shrink = []
    poly_4_score_shrink = []
    poly_4_score_no_shrink = []
    poly_5_score_shrink = []
    poly_5_score_no_shrink = []
    poly_10_score_shrink = []
    poly_10_score_no_shrink = []
    linear_time_shrink = []
    linear_time_no_shrink = []
    poly_3_time_shrink = []
    poly_3_time_no_shrink = []
    poly_4_time_shrink = []
    poly_4_time_no_shrink = []
    poly_5_time_shrink = []
    poly_5_time_no_shrink = []
    poly_10_time_shrink = []
    poly_10_time_no_shrink = []

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Linear Kernel - '
        f'Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='linear',
            C=c,
            shrinking=True,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        linear_score = scores.mean()
        linear_scores_shrink.append(linear_score)

        end = time.perf_counter()
        tuning_time = end - start
        linear_time_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {linear_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Linear Kernel - '
        f'No Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='linear',
            C=c,
            shrinking=False,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        linear_score = scores.mean()
        linear_scores_no_shrink.append(linear_score)

        end = time.perf_counter()
        tuning_time = end - start
        linear_time_no_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {linear_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 3rd Degree - '
        f'Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=3,
            C=c,
            shrinking=True,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_3_score = scores.mean()
        poly_3_score_shrink.append(poly_3_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_3_time_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_3_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 3rd Degree - '
        f'No Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=3,
            C=c,
            shrinking=False,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_3_score = scores.mean()
        poly_3_score_no_shrink.append(poly_3_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_3_time_no_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_3_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 4th Degree - '
        f'Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=4,
            C=c,
            shrinking=True,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_4_score = scores.mean()
        poly_4_score_shrink.append(poly_4_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_4_time_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_4_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 4th Degree - '
        f'No Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=4,
            C=c,
            shrinking=False,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_4_score = scores.mean()
        poly_4_score_no_shrink.append(poly_4_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_4_time_no_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_4_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 5th Degree - '
        f'Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=5,
            C=c,
            shrinking=True,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_5_score = scores.mean()
        poly_5_score_shrink.append(poly_5_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_5_time_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_5_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 5th Degree - '
        f'No Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=5,
            C=c,
            shrinking=False,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_5_score = scores.mean()
        poly_5_score_no_shrink.append(poly_5_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_5_time_no_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_5_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 10th Degree - '
        f'Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=10,
            C=c,
            shrinking=True,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_10_score = scores.mean()
        poly_10_score_shrink.append(poly_10_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_10_time_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_10_score} - Time: {tuning_time}')

    log_out(
        True,
        file,
        f'Tuning SVM for {title} with Polynomial Kernel, 10th Degree - '
        f'No Shrinking')

    for c in c_values:
        start = time.perf_counter()
        svm = SVC(
            kernel='poly',
            degree=10,
            C=c,
            shrinking=False,
            random_state=random_seed)
        scores = cross_val_score(svm, X=x, y=y, cv=folds)
        poly_10_score = scores.mean()
        poly_10_score_no_shrink.append(poly_10_score)

        end = time.perf_counter()
        tuning_time = end - start
        poly_10_time_no_shrink.append(tuning_time)
        log_out(
            True,
            file,
            f'C: {c} Accuracy: {poly_10_score} - Time: {tuning_time}')

    liner_shrink_max = max(linear_scores_shrink)
    linear_shrink_index = linear_scores_shrink.index(liner_shrink_max)
    liner_no_shrink_max = max(linear_scores_no_shrink)
    linear_no_shrink_index = linear_scores_no_shrink.index(liner_no_shrink_max)

    poly_3_shrink_max = max(poly_3_score_shrink)
    poly_3_shrink_ind = poly_3_score_shrink.index(poly_3_shrink_max)
    poly_3_no_shrink_max = max(poly_3_score_no_shrink)
    poly_3_no_shrink_ind = poly_3_score_no_shrink.index(poly_3_no_shrink_max)

    poly_4_shrink_max = max(poly_4_score_shrink)
    poly_4_shrink_ind = poly_4_score_shrink.index(poly_4_shrink_max)
    poly_4_no_shrink_max = max(poly_4_score_no_shrink)
    poly_4_no_shrink_ind = poly_4_score_no_shrink.index(poly_4_no_shrink_max)

    poly_5_shrink_max = max(poly_5_score_shrink)
    poly_5_shrink_ind = poly_5_score_shrink.index(poly_5_shrink_max)
    poly_5_no_shrink_max = max(poly_5_score_no_shrink)
    poly_5_no_shrink_ind = poly_5_score_no_shrink.index(poly_5_no_shrink_max)

    poly_10_shrink_max = max(poly_10_score_shrink)
    poly_10_shrink_ind = poly_10_score_shrink.index(poly_10_shrink_max)
    poly_10_no_shrink_max = max(poly_10_score_no_shrink)
    poly_10_no_shrink_ind = poly_10_score_no_shrink.index(
        poly_10_no_shrink_max)

    log_out(
        True,
        file,
        f'Kernel: linear - shrink - Score: {liner_shrink_max} '
        f'with c value {c_values[linear_shrink_index]}')
    log_out(
        True,
        file,
        f'Kernel: linear - No shrink - Score: {liner_no_shrink_max} '
        f'with c value {c_values[linear_no_shrink_index]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 3 - shrink - Score: {poly_3_shrink_max} '
        f'with c value {c_values[poly_3_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 3 - No shrink - Score: {poly_3_no_shrink_max}'
        f' with c value {c_values[poly_3_no_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 4 - shrink - Score: {poly_4_shrink_max} '
        f'with c value {c_values[poly_4_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 4 - No shrink - Score: {poly_4_no_shrink_max}'
        f' with c value {c_values[poly_4_no_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 5 - shrink - Score: {poly_5_shrink_max} '
        f'with c value {c_values[poly_5_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 5 - No shrink - Score: {poly_5_no_shrink_max}'
        f' with c value {c_values[poly_5_no_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 10 - shrink - Score: {poly_10_shrink_max} '
        f'with c value {c_values[poly_10_shrink_ind]}')
    log_out(
        True,
        file,
        f'Kernel: Poly - Degree: 10 - No shrink - Score: '
        f'{poly_10_no_shrink_max} with c value '
        f'{c_values[poly_10_no_shrink_ind]}')

    x = np.arange(len(c_values))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, linear_scores_shrink, width, label='shrink')
    ax.bar(x + width / 2, linear_scores_no_shrink, width, label='No shrink')
    ax.set_xticks(x, c_values)
    plt.title(f'SVM Tuning for {title} (Linear Kernel')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-SVM-Tuning-Linear.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(x - width/2, poly_3_score_shrink, width, label='shrink')
    ax.bar(x + width / 2, poly_3_score_no_shrink, width, label='No shrink')
    ax.set_xticks(x, c_values)
    plt.title(f'SVM Tuning for {title} (Poly Kernel Degree-3')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-SVM-Tuning-Poly-3.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(x - width/2, poly_4_score_shrink, width, label='shrink')
    ax.bar(x + width / 2, poly_4_score_no_shrink, width, label='No shrink')
    ax.set_xticks(x, c_values)
    plt.title(f'SVM Tuning for {title} (Poly Kernel Degree-4')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-SVM-Tuning-Poly-4.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(x - width/2, poly_5_score_shrink, width, label='shrink')
    ax.bar(x + width / 2, poly_5_score_no_shrink, width, label='No shrink')
    ax.set_xticks(x, c_values)
    plt.title(f'SVM Tuning for {title} (Poly Kernel Degree-5')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-SVM-Tuning-Poly-5.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(x - width/2, poly_10_score_shrink, width, label='shrink')
    ax.bar(x + width / 2, poly_10_score_no_shrink, width, label='No shrink')
    ax.set_xticks(x, c_values)
    plt.title(f'SVM Tuning for {title} (Poly Kernel Degree-10')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-SVM-Tuning-Poly-10.png')
    plt.close()


def neural_network_tune(x, y, folds, title, scale):
    cwd = getcwd()

    if scale:
        scaler = StandardScaler()
        scaler.fit(x)
        x_val = scaler.transform(x)
        file = open(f'{cwd}/notes/tuning/{title}-NN-tune-scaled.txt', 'w')
    else:
        x_val = x
        file = open(f'{cwd}/notes/tuning/{title}-NN-tune.txt', 'w')

    neuron_range = range(1, 201, 2)
    neuron_values = []

    learning_rate_001 = 0.001
    results_001 = []

    learning_rate_01 = 0.01
    results_01 = []

    learning_rate_0001 = 0.0001
    results_0001 = []

    log_out(
        True,
        file,
        f'Tuning NN for {title} with learning rate of {learning_rate_001}')
    for i in neuron_range:
        neuron_values.append(i)
        hidden_layer_size = (1, i)
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            max_iter=5000,
            learning_rate_init=learning_rate_001,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x_val,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'Neurons {i}: {mean_score}')
        results_001.append(mean_score)

    log_out(
        True,
        file,
        f'Tuning NN for {title} with learning rate of {learning_rate_0001}')
    for i in neuron_range:
        hidden_layer_size = (1, i)
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            max_iter=5000,
            learning_rate_init=learning_rate_0001,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x_val,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'Neurons {i}: {mean_score}')
        results_0001.append(mean_score)

    log_out(
        True,
        file,
        f'Tuning NN for {title} with learning rate of {learning_rate_01}')
    for i in neuron_range:
        hidden_layer_size = (1, i)
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            max_iter=5000,
            learning_rate_init=learning_rate_01,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x_val,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'Neurons {i}: {mean_score}')
        results_01.append(mean_score)

    rate_01_max = max(results_01)
    rate_01_max_ind = results_01.index(rate_01_max)
    rate_001_max = max(results_001)
    rate_001_max_ind = results_001.index(rate_001_max)
    rate_0001_max = max(results_0001)
    rate_0001_max_ind = results_0001.index(rate_0001_max)

    log_out(
        True,
        file,
        f'Learning rate 0.01: Best Accuracy: {rate_01_max} '
        f'with neuron count of {neuron_values[rate_01_max_ind]}')
    log_out(
        True,
        file,
        f'Learning rate 0.001: Best Accuracy: {rate_001_max} '
        f'with neuron count of {neuron_values[rate_001_max_ind]}')
    log_out(
        True,
        file,
        f'Learning rate 0.0001: Best Accuracy: {rate_0001_max} '
        f'with neuron count of {neuron_values[rate_0001_max_ind]}')

    fig, ax = plt.subplots()
    plt.title(f'NN Tuning for {title}')
    plt.xlabel('Neurons')
    plt.ylabel('Cross Validation Accuracy')
    ax.plot(
        neuron_values,
        results_01,
        label='Learning Rate 0.01',
        color='red')
    ax.plot(
        neuron_values,
        results_001,
        label='Learning Rate 0.001',
        color='blue')
    ax.plot(
        neuron_values,
        results_0001,
        label='Learning Rate 0.0001',
        color='green')
    ax.legend(loc='lower right')
    if scale:
        plt.savefig(f'./images/tuning/{title}-NN-Tuning-scale.png')
    else:
        plt.savefig(f'./images/tuning/{title}-NN-Tuning.png')
    plt.close()

    file.close()


def decision_tree_tune(x, y, max_levels, folds, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/tuning/{title}-DT-tune.txt', 'w')

    lvl_range = range(1, max_levels+1)

    min_leaf_1_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 1')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=1,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_1_results.append(scores.mean())

    min_leaf_2_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 2')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=2,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_2_results.append(scores.mean())

    min_leaf_3_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 3')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=3,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_3_results.append(scores.mean())

    min_leaf_4_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 4')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=4,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_4_results.append(scores.mean())

    min_leaf_5_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 5')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=5,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_5_results.append(scores.mean())

    min_leaf_10_results = []
    log_out(
        True,
        file,
        f'Tuning DT for {title} with max depth of {max_levels} '
        'using a min leaf value of 10')
    for i in lvl_range:
        classifier = DecisionTreeClassifier(
            max_depth=i,
            criterion='gini',
            min_samples_leaf=10,
            random_state=random_seed)
        scores = cross_val_score(
            estimator=classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{i}: {mean_score}')
        min_leaf_10_results.append(scores.mean())

    min_leaf_1_max = max(min_leaf_1_results)
    min_leaf_1_ind = min_leaf_1_results.index(min_leaf_1_max)
    min_leaf_2_max = max(min_leaf_2_results)
    min_leaf_2_ind = min_leaf_2_results.index(min_leaf_2_max)
    min_leaf_3_max = max(min_leaf_3_results)
    min_leaf_3_ind = min_leaf_3_results.index(min_leaf_3_max)
    min_leaf_4_max = max(min_leaf_4_results)
    min_leaf_4_ind = min_leaf_4_results.index(min_leaf_4_max)
    min_leaf_5_max = max(min_leaf_5_results)
    min_leaf_5_ind = min_leaf_5_results.index(min_leaf_5_max)
    min_leaf_10_max = max(min_leaf_10_results)
    min_leaf_10_ind = min_leaf_10_results.index(min_leaf_10_max)

    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 1 is '
        f'{min_leaf_1_max} with depth of {lvl_range[min_leaf_1_ind]}')
    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 2 is '
        f'{min_leaf_2_max} with depth of {lvl_range[min_leaf_2_ind]}')
    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 3 is '
        f'{min_leaf_3_max} with depth of {lvl_range[min_leaf_3_ind]}')
    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 4 is '
        f'{min_leaf_4_max} with depth of {lvl_range[min_leaf_4_ind]}')
    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 5 is '
        f'{min_leaf_5_max} with depth of {lvl_range[min_leaf_5_ind]}')
    log_out(
        True,
        file,
        f'Best accyracy for min sample split of 10 is '
        f'{min_leaf_10_max} with depth of {lvl_range[min_leaf_10_ind]}')

    fig, ax = plt.subplots()
    plt.title(f'DT Tuning for {title}')
    plt.xlabel('Max Depth')
    plt.ylabel('Cross Validation Accuracy')
    ax.plot(lvl_range, min_leaf_1_results, label='Min 1', color='black')
    ax.plot(lvl_range, min_leaf_2_results, label='Min 2', color='red')
    ax.plot(lvl_range, min_leaf_3_results, label='Min 3', color='purple')
    ax.plot(lvl_range, min_leaf_4_results, label='Min 4', color='orange')
    ax.plot(lvl_range, min_leaf_5_results, label='Min 5', color='green')
    ax.plot(lvl_range, min_leaf_10_results, label='Min 10', color='blue')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-DT-Tuning.png')
    plt.close()
    file.close()


def knn_tune(x, y, max_k, folds, title):
    cwd = getcwd()
    file = open(f'{cwd}/notes/tuning/{title}-KNN-tune.txt', 'w')

    k_range = range(1, max_k+1)
    k_value = []
    uniform_results_manhattan = []
    uniform_results_euclidian = []
    weighted_results_manhattan = []
    weighted_results_euclidian = []

    log_out(
        True,
        file,
        f'Tuning KNN for {title} with Uniform Weights and Manhattan Distance')
    for k in k_range:
        k_value.append(k)
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='uniform',
            p=1)
        scores = cross_val_score(knn, X=x, y=y, cv=folds)
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{k}: {mean_score}')
        uniform_results_manhattan.append(mean_score)

    log_out(
        True,
        file,
        f'Tuning KNN for {title} with Uniform Weights and Euclidean Distance')
    for k in k_range:
        k_value.append(k)
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='uniform',
            p=2)
        scores = cross_val_score(knn, X=x, y=y, cv=folds)
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{k}: {mean_score}')
        uniform_results_euclidian.append(mean_score)

    log_out(
        True,
        file,
        f'Tuning KNN for {title} with Manhattan Distance Weights')
    for k in k_range:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            p=1)
        scores = cross_val_score(knn, X=x, y=y, cv=folds)
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{k}: {mean_score}')
        weighted_results_manhattan.append(mean_score)

    log_out(
        True,
        file,
        f'Tuning KNN for {title} with Manhattan Distance Weights')
    for k in k_range:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            p=1)
        scores = cross_val_score(knn, X=x, y=y, cv=folds)
        mean_score = scores.mean()
        log_out(
            True,
            file,
            f'{k}: {mean_score}')
        weighted_results_euclidian.append(mean_score)

    uniform_max_man = max(uniform_results_manhattan)
    uniform_max_man_index = uniform_results_manhattan.index(uniform_max_man)
    uniform_max_euc = max(uniform_results_euclidian)
    uniform_max_euc_index = uniform_results_euclidian.index(uniform_max_euc)
    weighted_max_man = max(weighted_results_manhattan)
    weighted_max_man_index = weighted_results_manhattan.index(weighted_max_man)
    weighted_max_euc = max(weighted_results_euclidian)
    weighted_max_euc_index = weighted_results_euclidian.index(weighted_max_euc)

    log_out(
        True,
        file,
        f'Best uniform max (Man): is  k: {k_value[uniform_max_man_index]} '
        f'with accuracy of {uniform_max_man}')
    log_out(
        True,
        file,
        f'Best uniform max (Euc): is  k: {k_value[uniform_max_euc_index]} '
        f'with accuracy of {uniform_max_euc}')
    log_out(
        True,
        file,
        f'Best weighted max (Man): is k: {k_value[weighted_max_man_index]} '
        f'with accuracy of {weighted_max_man}')
    log_out(
        True,
        file,
        f'Best weighted max (Euc): is k: {k_value[weighted_max_euc_index]} '
        f'with accuracy of {weighted_max_euc}')

    fig, ax = plt.subplots()
    plt.title(f'KNN Tuning for {title}')
    plt.xlabel('K')
    plt.ylabel('Cross Validation Accuracy')

    ax.plot(
        k_range,
        uniform_results_manhattan,
        label='Uniform (Man)',
        color='red')
    ax.plot(
        k_range,
        uniform_results_euclidian,
        label='Uniform (Euc)',
        color='blue')
    ax.plot(
        k_range,
        weighted_results_manhattan,
        label='Weighted (Man)',
        color='green')
    ax.plot(
        k_range,
        weighted_results_euclidian,
        label='Weighted (Euc)',
        color='black')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-KNN-Tuning.png')
    plt.close()


def boosting_tune(x, y, title, depth, min_samples, folds):
    cwd = getcwd()
    file = open(f'{cwd}/notes/tuning/{title}-BOOST-tune.txt', 'w')

    learning_rates = [1, 0.1, 0.01, 0.001]
    estimator = 2500

    unpruned_scores = []
    pruned_scores = []

    tree_unpruned = DecisionTreeClassifier(
            max_depth=depth,
            criterion='gini',
            min_samples_leaf=min_samples,
            random_state=random_seed)

    tree_pruned = DecisionTreeClassifier(
            max_depth=2,
            criterion='gini',
            min_samples_leaf=min_samples,
            random_state=random_seed)

    for rate in learning_rates:
        log_out(
            True,
            file,
            f'Tuning Boosting for {title} with learning rate of {rate} '
            ' for unpruned tree')
        unpruned_classifier = AdaBoostClassifier(
            n_estimators=estimator,
            estimator=tree_unpruned,
            learning_rate=rate)

        scores = cross_val_score(
            estimator=unpruned_classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')

        mean_score = scores.mean()
        unpruned_scores.append(mean_score)
        log_out(
            True,
            file,
            f'Learning rate {rate} for unpruned produced: {mean_score}')

        pruned_classifier = AdaBoostClassifier(
            n_estimators=estimator,
            estimator=tree_pruned,
            learning_rate=rate)

        scores = cross_val_score(
            estimator=pruned_classifier,
            X=x,
            y=y,
            cv=folds,
            n_jobs=4,
            error_score='raise')

        mean_score = scores.mean()
        pruned_scores.append(mean_score)
        log_out(
            True,
            file,
            f'Learning rate {rate} for pruned produced: {mean_score}')

    max_unpruned = max(unpruned_scores)
    max_unpruned_ind = unpruned_scores.index(max_unpruned)
    max_pruned = max(pruned_scores)
    max_pruned_ind = pruned_scores.index(max_pruned)

    log_out(
        True,
        file,
        f'Best accuracy for unpruned tree was {max_unpruned} with learning '
        f'rate of {learning_rates[max_unpruned_ind]}')
    log_out(
        True,
        file,
        f'Best accuracy for pruned tree was {max_pruned} with learning '
        f'rate of {learning_rates[max_pruned_ind]}')

    x = np.arange(len(learning_rates))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, unpruned_scores, width, label='Unpruned')
    ax.bar(x + width / 2, pruned_scores, width, label='Pruned')
    ax.set_xticks(x, learning_rates)
    plt.title(f'Boost Tuning for {title}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Cross Validation Accuracy')
    ax.legend(loc='lower right')
    plt.savefig(f'./images/tuning/{title}-Boost-Tuning.png')
    plt.close()
