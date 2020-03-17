import pandas as pd
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import ml_challenge_options as options_file
my_dpi = 96


class ClassificationTraining(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def grid_search(self, parameters, k, score):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=k, scoring=score, n_jobs=-1, error_score=np.nan)

    def clf_grid_fit(self, x, y):
        self.grid.fit(x, y)

    def predict(self, x):
        self.grid.predict(x)

    def feature_importance(self):
        return self.clf.feature_importances_


class ClassificationEvaluation(object):
    def __init__(self, groundtruth, prediction):
        self.f1_micro = f1_score(groundtruth, np.round(prediction), average='micro', labels=np.unique(np.round(prediction)), zero_division=0)
        self.f1_average = f1_score(groundtruth, np.round(prediction), average='weighted', labels=np.unique(np.round(prediction)), zero_division=0)
        self.f1_macro = f1_score(groundtruth, np.round(prediction), average='macro', labels=np.unique(np.round(prediction)), zero_division=0)
        self.accuracy = accuracy_score(groundtruth, np.round(prediction))
        self.precision = precision_score(groundtruth, np.round(prediction), average=None)
        self.recall = recall_score(groundtruth, np.round(prediction), average=None)

        self.classification_report = classification_report(groundtruth, np.round(prediction))

        self.available_metrics = ['f1_micro', 'f1_average', 'f1_macro', 'accuracy', 'precision', 'recall']
        self.roc_auc_curve = roc_auc_score(groundtruth, np.round(prediction))  # Note: this implementation is restricted to the binary classification task or multilabel classification task in label indicator format.


def main():
    file_loc = options_file.df_original_path
    df = data_retrieval(file_loc)
    x_train, y_train, x_test, y_test = data_processing(df)
    trained_models_dict, classes = data_modelling(x_train, y_train)
    data_evaluation(trained_models_dict, classes, x_train, y_train, x_test, y_test)


def data_retrieval(file_loc):
    df = pd.read_csv(file_loc)

    return df


def data_processing(dataset):
    dataset.drop(['N2'], axis=1, inplace=True)  # N2 possui demasiados valores diferentes assim como uma correlação alta com N1;
    dataset['LABEL'] = dataset['LABEL'] - 1  # Tipicamente modelos comportam-se melhor com labels binárias;

    grouping_dictionaries = [options_file.c2_dict, options_file.c4_dict, options_file.c5_1_dict, options_file.c9_dict, options_file.c10_dict]
    dataset = col_group(dataset, ['C2', 'C4', 'C5.1', 'C9', 'C10'], grouping_dictionaries)

    dataset.to_csv(options_file.df_before_dummies_path)  # Passo necessário para utilização na app

    # Variáveis Categóricas - One Hot Encoding:
    dataset = pd.get_dummies(dataset)

    dataset.to_csv(options_file.df_processed_path)  # Passo necessário para utilização na app

    train_dataset, test_dataset = train_test_split(dataset, train_size=0.7, stratify=dataset['LABEL'])

    ros = RandomOverSampler(random_state=42)
    x_train, y_train = train_dataset[[x for x in list(train_dataset) if x != 'LABEL']], train_dataset['LABEL']
    train_x_resampled, train_y_resampled = ros.fit_sample(x_train, y_train.values.ravel())

    x_test, y_test = test_dataset[[x for x in list(test_dataset) if x != 'LABEL']], test_dataset['LABEL']
    return train_x_resampled, train_y_resampled, x_test, y_test


def data_modelling(x_train, y_train):
    models = options_file.classification_models.keys()
    model_dict = {}

    for model in models:
        print('Training model {}...'.format(model))
        clf = ClassificationTraining(clf=options_file.classification_models[model][0])
        clf.grid_search(parameters=options_file.classification_models[model][1], k=options_file.k, score=options_file.gridsearch_score)
        clf.clf_grid_fit(x=x_train, y=y_train)
        clf_best = clf.grid.best_estimator_

        model_dict[model] = clf_best
        save_model(clf_best, model)

    classes = clf.grid.classes_
    return model_dict, classes


def data_evaluation(trained_models_dict, classes, x_train, y_train, x_test, y_test):
    models = trained_models_dict.keys()

    results_train, results_test = [], []
    for model in models:
        print('Evaluating model {}...'.format(model))
        start_prediction_time_train = time.time()
        prediction_train = trained_models_dict[model].predict(x_train)
        end_prediction_time_train = time.time()

        start_prediction_time_test = time.time()
        prediction_test = trained_models_dict[model].predict(x_test)
        end_prediction_time_test = time.time()

        plot_conf_matrix(trained_models_dict[model], model, x_test, y_test, classes)

        evaluation_training = ClassificationEvaluation(groundtruth=y_train, prediction=prediction_train)
        evaluation_test = ClassificationEvaluation(groundtruth=y_test, prediction=prediction_test)

        row_train = {'Micro_F1': getattr(evaluation_training, 'f1_micro'),
                     'Average_F1': getattr(evaluation_training, 'f1_average'),
                     'Macro_F1': getattr(evaluation_training, 'f1_macro'),
                     'Accuracy': getattr(evaluation_training, 'accuracy'),
                     'ROC_Curve': getattr(evaluation_training, 'roc_auc_curve'),
                     ('Precision_Class_' + str(classes[0])): getattr(evaluation_training, 'precision')[0],
                     ('Precision_Class_' + str(classes[1])): getattr(evaluation_training, 'precision')[1],
                     ('Recall_Class_' + str(classes[0])): getattr(evaluation_training, 'recall')[0],
                     ('Recall_Class_' + str(classes[1])): getattr(evaluation_training, 'recall')[1],
                     'Running_Time': round(end_prediction_time_train - start_prediction_time_train, 3)}

        row_test = {'Micro_F1': getattr(evaluation_test, 'f1_micro'),
                    'Average_F1': getattr(evaluation_test, 'f1_average'),
                    'Macro_F1': getattr(evaluation_test, 'f1_macro'),
                    'Accuracy': getattr(evaluation_test, 'accuracy'),
                    'ROC_Curve': getattr(evaluation_test, 'roc_auc_curve'),
                    ('Precision_Class_' + str(classes[0])): getattr(evaluation_test, 'precision')[0],
                    ('Precision_Class_' + str(classes[1])): getattr(evaluation_test, 'precision')[1],
                    ('Recall_Class_' + str(classes[0])): getattr(evaluation_test, 'recall')[0],
                    ('Recall_Class_' + str(classes[1])): getattr(evaluation_test, 'recall')[1],
                    'Running_Time': round(end_prediction_time_test - start_prediction_time_test, 3)}

        results_train.append(row_train)
        results_test.append(row_test)

    df_results_train = pd.DataFrame(results_train, index=models)
    df_results_train['Dataset'] = ['Train'] * df_results_train.shape[0]
    df_results_test = pd.DataFrame(results_test, index=models)
    df_results_test['Dataset'] = ['Test'] * df_results_train.shape[0]

    metric_bar_plot(df_results_train, classes, 'train')
    metric_bar_plot(df_results_test, classes, 'test')


def metric_bar_plot(df, classes, tag):
    algorithms = df.index
    algorithms_count = len(algorithms)

    i, j = 0, 0
    metrics_available = ['Micro_F1', 'Average_F1', 'Macro_F1', 'Accuracy', 'ROC_Curve'] + ['Precision_Class_{}'.format(x) for x in classes] + ['Recall_Class_{}'.format(x) for x in classes] + ['Running_Time']

    fig, ax = plt.subplots(2, 5, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    for metric in metrics_available:
        ax[j, i].bar(range(0, algorithms_count), df[metric].values)
        ax[j, i].set_title(metric)
        ax[j, i].grid()
        plt.setp(ax[j, i].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax[j, i], xticks=range(0, algorithms_count), xticklabels=algorithms)
        if metric != 'Running_Time':
            ax[j, i].set_ylim(0, 1.01)
        k = 0
        for value in df[metric].values:
            ax[j, i].text(k - 0.45, round(value, 2) * 1.01, '{:.2f}'.format(value), color='red')
            k += 1

        i += 1
        if i == 5:
            i = 0
            j += 1

    plt.tight_layout()
    plt.savefig(options_file.plots_path + 'performance_' + tag)
    # plt.show()


def col_group(df, columns_to_replace, dictionaries):
    for column, column_dictionary in zip(columns_to_replace, dictionaries):
        for key in column_dictionary.keys():
            df.loc[df[column].isin(column_dictionary[key]), column] = key

    return df


def plot_conf_matrix(clf, model_name, x_test, y_test, classes):

    plot_confusion_matrix(clf, x_test, y_test, display_labels=classes, cmap=plt.cm.Blues)
    plt.savefig(options_file.plots_path + '{}_confusion_matrix'.format(model_name))
    plt.show()


def save_model(model, model_name):

    file_name = options_file.models_path + '{}_model.sav'.format(model_name)
    file_handler = open(file_name, 'wb')
    pickle.dump(model, file_handler)
    file_handler.close()

    return


if __name__ == '__main__':
    main()
