# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:26:36 2021

@author: Pedro.Oliveira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from matplotlib.gridspec import GridSpec
from matplotlib import cm

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, learning_curve
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            precision_recall_curve, f1_score, confusion_matrix, roc_curve, \
                            roc_auc_score

from IPython.display import HTML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
import time
import warnings
warnings.filterwarnings('ignore')


def format_spines(ax, right_border=True):
    """
    This function sets up borders from an axis and personalize colors

    Input:
        Axis and a flag for deciding or not to plot the right border
    Returns:
        Plot configuration
    """
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')

def categorical_plot(cols_cat, axs, df):
    """
    this function receives a list of categorical features and plot all of them in a 1, 2 grid.

    input:
        list of categorical features
    returns:
        categorical feature plot
    """

    idx_row = 0
    for col in cols_cat:
        # Returning column index
        idx_col = cols_cat.index(col)

        # Verifying brake line in figure (second row)
        if idx_col >= 2:
            idx_col -= 2
            idx_row = 0

        # Plot params
        names = df[col].value_counts().index
        heights = df[col].value_counts().values

        # Bar chart
        axs[idx_col].bar(names, heights, color='navy')
        total = df[col].value_counts().sum()
        axs[idx_col].patch.set_facecolor('#FFFFFF')
        format_spines(axs[idx_col], right_border=False)
        for p in axs[idx_col].patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            axs[idx_col].annotate('{:.1%}'.format(h/1000), (p.get_x()+.29*w,
                                            p.get_y()+h+20), color='k')

        # Plot configuration
        axs[idx_col].set_title(col, size=12)
        axs[idx_col].set_ylim(0, heights.max()+120)

def individual_cat_pie_plot(col, ax, cs, df):

    """this function plot a pie chart of a categorical attribute

    input:
        categorical feature
    returns:
        singular plot"""

    # Creating figure and showing data
    names = df[col].value_counts().index
    heights = df[col].value_counts().values
    total = df[col].value_counts().sum()
    #if cs:
    #cs = cm.viridis(np.arange(len(names))/len(names))
    explode = np.zeros(len(names))
    explode[0] = 0.05
    wedges, texts, autotexts = ax.pie(heights, labels=names, explode=explode,
                                       startangle=90, shadow=False,
                                      autopct='%1.1f%%', colors=cs[:len(names)])
    plt.setp(autotexts, size=12, color='w')

def donut_plot(col, ax, df, text='', colors=['navy', 'crimson'],
               labels=['good', 'bad']):
    """
    this function plots a customized donut chart
    """
    sizes = df[col].value_counts().values
    #labels = df[col].value_counts().index
    center_circle = plt.Circle((0,0), 0.80, color='white')
    ax.pie((sizes[0], sizes[1]), labels=labels, colors=colors, autopct='%1.1f%%')
    ax.add_artist(center_circle)
    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)

def plot_roc_curve(fpr, tpr, label=None):
    """
    this function plots the ROC curve of a model

    input:
        fpr: false positive rate
        tpr: true positive rate
    returns:
        ROC curve
    """

    # Showing data
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function set up and plot a Confusion Matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format plot
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_dataset():
    """
    This functions creates a dataframe to keep performance analysis
    """
    attributes = ['acc', 'prec', 'rec', 'f1', 'auc', 'total_time']
    model_performance = pd.DataFrame({})
    for col in attributes:
        model_performance[col] = []
    return model_performance

def model_analysis(classifiers, X, y, df_performance, cv=5, train=True):
    """
    This function brings up a full model evaluation and saves it in a DataFrame object.
    """
    for key, model in classifiers.items():
        t0 = time.time()

        # Accuracy, precision, recall and f1_score on training set using cv
        if train:
            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
            prec = cross_val_score(model, X, y, cv=cv, scoring='precision').mean()
            rec = cross_val_score(model, X, y, cv=cv, scoring='recall').mean()
            f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
        else:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)

        # AUC score
        try:
            y_scores = cross_val_predict(model, X, y, cv=5,
                                     method='decision_function')
        except:
            # Trees don't have decision_function but predict_proba
            y_probas = cross_val_predict(model, X, y, cv=5,
                                         method='predict_proba')
            y_scores_tree = y_probas[:, 1]
            y_scores = y_scores_tree
        auc = roc_auc_score(y, y_scores)

        t1 = time.time()
        delta_time = t1-t0
        model_name = model.__class__.__name__

        # Saving on dataframe
        performances = {}
        performances['acc'] = round(acc, 4)
        performances['prec'] = round(prec, 4)
        performances['rec'] = round(rec, 4)
        performances['f1'] = round(f1, 4)
        performances['auc'] = round(auc, 4)
        performances['total_time'] = round(delta_time, 3)

        df_performance = df_performance.append(performances, ignore_index=True)
    df_performance.index = classifiers.keys()

    return df_performance

def model_confusion_matrix(classifiers, X, y, cmap=plt.cm.Blues):
    """
    This function computes predictions for all model and plots a confusion matrix
    for each one.
    """
    i = 1
    plt.figure(figsize=(11, 11))
    sns.set(style='white', palette='muted', color_codes=True)
    labels = ['Good', 'Bad']

    # Ploting confusion matrix
    for key, model in classifiers.items():
        y_pred = model.predict(X)
        model_cf_mx = confusion_matrix(y, y_pred)

        # Plotando matriz
        model_name = model.__class__.__name__
        plt.subplot(3, 3, i)
        plot_confusion_matrix(model_cf_mx, labels, title=model_name + '\nConfusion Matrix', cmap=cmap)
        i += 1

    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, y, y_scores, auc, label=None):
    """
    This function plots the ROC curve of a model
    """
    # Showing data
    sns.set(style='white', palette='muted', color_codes=True)
    plt.plot(fpr, tpr, linewidth=2, label=f'{label}, auc={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.02, 1.02, -0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve', size=14)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)',
                 xy=(0.5, 0.5), xytext=(0.6, 0.4),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

def plot_precision_vs_recall(precisions, recalls, label=None, color='b'):
    """
    This function plots precision versus recall curve.
    """
    sns.set(style='white', palette='muted', color_codes=True)
    if label=='LogisticRegression':
        plt.plot(recalls, precisions, 'r-', linewidth=2, label=label)
    else:
        plt.plot(recalls, precisions, color=color, linewidth=2, label=label)
    plt.title('Precision versus Recall', fontsize=14)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.legend()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    This function plots precision x recall among different thresholds
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.title('Precision versus Recall - Thresholds', size=14)
    plt.legend(loc='best')
    plt.ylim([0, 1])

def plot_learning_curve(trained_models, X, y, ylim=None, cv=5, n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 10)):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(17, 17))
    if ylim is not None:
        plt.ylim(*ylim)
    i = 0
    j = 0
    for key, model in trained_models.items():
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        axs[i, j].fill_between(train_sizes, train_scores_mean - train_scores_std,
                               train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        axs[i, j].fill_between(train_sizes, test_scores_mean - test_scores_std,
                               test_scores_mean + test_scores_std, alpha=0.1, color='crimson')
        axs[i, j].plot(train_sizes, train_scores_mean, 'o-', color="navy",
                       label="Training score")
        axs[i, j].plot(train_sizes, test_scores_mean, 'o-', color="red",
                       label="Cross-Validation score")
        axs[i, j].set_title(f'{key} Learning Curve', size=14)
        axs[i, j].set_xlabel('Training size (m)')
        axs[i, j].set_ylabel('Score')
        axs[i, j].grid(True)
        axs[i, j].legend(loc='best')
        j += 1
        if j == 2:
            i += 1
            j = 0

def keep_the_first_default(data):
    """
    Keep the first default in the data set
    """
    data_modele = data.loc[(data.MODELE == 3) + (data.MODELE == 4)]
    col = ['ID', 'RATING','ANCIENNETE', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
           'CIBLE']
    df_modele = data_modele.loc[:,col].sort_index()
    df = pd.DataFrame(columns = col)
    for i, df_g in df_modele.groupby('ID'):
        keep = (df_g.duplicated() == False)
        df_g = df_g.loc[keep]
        num = []
        controller = 0
        for j in range(df_g.shape[0]):
            if df_g.CIBLE[j] == 1:
                if controller == 0:
                    num.append(j)
                controller = 1
            else:
                num.append(j)
                controller = 0
        df_g = df_g.iloc[num,:]
        df_g.fillna(df_g.mean(), inplace=True)
        df_g.CIBLE = df_g.CIBLE.shift(1)
        df = df.append(df_g.iloc[1:])

    return df

def create_pipelines(numerical_attribs, categorical_attribs, categories):
    """
    Transform the data using the stadardized method for numerical values and 
    one-hot enconder for categorical values
    
    Parameters
    ----------
    numerical_attribs : DataFrame
        numerical attributes values.
    categorical_attribs : DataFrame
        categorical attributes values.
    categories : list of arrays
        List of categorical possible values.

    Returns
    -------
    full_pipeline : numpy matrix
        converted matrix.

    """
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('one_hot', OneHotEncoder(sparse=False, categories=categories))
    ])
    
    # full pipeline
    full_pipeline = ColumnTransformer([
        ('num_pipeline', numerical_pipeline, numerical_attribs),
        ('cat_pipeline', categorical_pipeline, categorical_attribs)
    ])
    
    return full_pipeline

def generate_dict_analysis(model, *kwargs):

    first_trial = {
        'improvement_name': 'First Trial',
        'model': trained_models[model],
        'model_name': model + ' - First Trial',
        'train_set': X_train_prepared,
        'y_train': y_train,
        'test_set': X_test_prepared,
        'y_test': y_test
    }

    improvement = {
        'improvement_name': 'Oversampling',
        'model': trained_models_imp3[model],
        'model_name': model + ' - Oversampling',
        'train_set': X_train_prepared_imp3,
        'y_train': y_train_res,
        'test_set': X_test_prepared_imp3,
        'y_test': y_test_imp3
    }

    roc_comparison = {
        'first_trial': first_trial,
        'improvement': improvement
    }

    return roc_comparison

class dict_analysis:

    def __init__(self, **kwargs):
        self.first_trial = {
            'improvement_name': 'Raw',
            'model': "",
            'model_name':"" ,
            'train_set': kwargs['train_set'],
            'y_train': kwargs['y_train'],
            'test_set': kwargs['test_set'],
            'y_test': kwargs['y_test']
        }

        self.improvement = {
            'improvement_name': 'Oversampling',
            'model': "",
            'model_name': "",
            'train_set': kwargs['train_set_imp'],
            'y_train': kwargs['y_train_imp'],
            'test_set': kwargs['test_set_imp'],
            'y_test': kwargs['y_test_imp']
        }
    
    def update_dict(self, model, raw_model, improved_model):
        self.first_trial['model'] = raw_model[model]
        self.first_trial['model_name'] = model + ' - First Trial'
        self.improvement['model'] = improved_model[model]
        self.improvement['model_name'] = model + ' - Oversampling'
    
    def generate_analysis(self):
        self.roc_comparison = {
            'first_trial': self.first_trial,
            'improvement': self.improvement
        }
        return self.roc_comparison