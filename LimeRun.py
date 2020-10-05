################
## Libraries
################
import pandas as pd
import numpy as np
import glob
import re
import lime
import random

import time
import tqdm ## Timer
# from tqdm._tqdm_notebook import tqdm_notebook ## Adding notebook support

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import shap

import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import functions
################
## Requirements
################
tqdm.pandas(desc = "my bar!") ## Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`



################
## Data Preparation
################
fileName1 = "/home/huiyangd/toxicSpans/data/2018_01_n_1000_32.csv"
fileName2 = "/home/huiyangd/toxicSpans/data/2018_01_p_1000_32.csv"

RC_2018_01_n_1000_32 = pd.read_csv(fileName1)
RC_2018_01_p_1000_32 = pd.read_csv(fileName2)
frames = [RC_2018_01_n_1000_32, RC_2018_01_p_1000_32]
RC_2018_01_combined_1000 = pd.concat(frames)

class_names = list(RC_2018_01_combined_1000.toxicity.unique())

X_train, X_test, y_train, y_test = train_test_split(RC_2018_01_combined_1000['body'],
                                                    RC_2018_01_combined_1000['toxicity'],
                                                    test_size = 0.25,
                                                    random_state = 32)
################
## Text Embedding and Split
################

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase = False)
train_vectors = vectorizer.fit_transform(X_train.to_list())
test_vectors = vectorizer.transform(X_test.to_list())


################
## Model Fitting
################
rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 500)
rf.fit(train_vectors, y_train)

lr = LogisticRegression(n_jobs = 1, C = 1e5)
lr.fit(train_vectors, y_train)

pred_rf = rf.predict(test_vectors)
pred_lr = lr.predict(test_vectors)
c_rf = make_pipeline(vectorizer, rf)
c_lr = make_pipeline(vectorizer, lr)
explainer = LimeTextExplainer(class_names=class_names)

functions.printing_Results(y_test, pred_rf)
functions.printing_Results(y_test, pred_lr)

## Lime
TestDF = pd.concat([X_test, y_test], axis = 1)
TestDF['LimeRF'] = TestDF.progress_apply(lambda row: make_Explanations(c_rf, explainer, row, num_features = 10), axis = 1)
TestDF['LimeLR'] = TestDF.progress_apply(lambda row: make_Explanations(c_lr, explainer, row, num_features = 10), axis = 1)
TestDF.to_csv("/home/huiyangd/toxicSpans/data/LimeLabeledData.csv")

# ## SHAP
# X_train_sample = shap.sample(train_vectors, 400)
# X_test_sample = shap.sample(test_vectors, 200)
# start_time = time.time()
# # SHAP_explainer = shap.KernelExplainer(rf.predict, X_train_sample)
# # shap_vals = SHAP_explainer.shap_values(X_test_sample)
# end_time = time.time()
# print("The processing time for SHAP is: %s seconds" % (end_time - start_time))