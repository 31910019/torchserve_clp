from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.lunar import LUNAR
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

from pathlib import Path
import pandas as pd
import os
sampling = '6S'
directory = os.path.join('F:/PycharmProjects/pythonProject/data', 'uk_dale')
directory = Path(directory)
house_id = 1
house_folder = directory.joinpath('house_' + str(house_id))
# house_folder = house_folder.joinpath('labels.dat').__str__().replace('\\','/')
house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)

house_data = pd.read_csv(house_folder.joinpath(
    'channel_1.dat'), sep=' ', header=None)

house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
house_data.columns = ['time', 'aggregate']
house_data = house_data.set_index('time')

house_data = house_data.resample(sampling).mean().fillna(
                    method='ffill', limit=30)

from collections import defaultdict

appliance_list = house_label.iloc[:, 1].values
app_index_dict = defaultdict(list)
appliance_names = appliance_list.tolist()
appliance_names = list(set(appliance_names))
appliance_names.remove('aggregate')

for appliance in appliance_names:
    data_found = False
    for i in range(len(appliance_list)):
        if appliance_list[i] == appliance:
            app_index_dict[appliance].append(i + 1)
            data_found = True

    if not data_found:
        app_index_dict[appliance].append(-1)


import numpy as np
# for appliance in appliance_names[:1]:
#     if app_index_dict[appliance][0] == -1:
#         house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
#     else:
#         temp_data = pd.read_csv(house_folder.joinpath(
#             'channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)
#         temp_data.iloc[:, 0] = pd.to_datetime(
#             temp_data.iloc[:, 0], unit='s')
#         temp_data.columns = ['time', appliance]
#         temp_data = temp_data.set_index('time')
#         temp_data = temp_data.resample(sampling).mean().fillna(
#                             method='ffill', limit=30)
#         house_data = pd.merge(
#             house_data, temp_data, how='left', on='time')
#
# imputed_house_data = house_data.fillna(method='ffill', limit=30)
# imputed_house_data.to_csv(os.path.join('data/aligned_ukdale/'
#     'house_' + str(house_id) + '_imputed5.csv'))
# imputed_house_data.resample('30S').mean().fillna(method='ffill', limit=30)
aggregate = house_data['aggregate']
aggregate[aggregate.isna()]
aggregate = aggregate.fillna(-10000)

import csv
# suppose s is your pandas Series
s = aggregate[:100000]
window_size = 480
batch_size = 1000  # adjust this to a size that fits comfortably in memory
with open(f'data/ukdale_aggreate_windowed_series_house_{house_id}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(window_size, len(s)+1, batch_size):
        batch = [s[j-window_size:j].tolist() for j in range(i, min(i+batch_size, len(s)+1))]
        writer.writerows(batch)

import pandas as pd
X_train = pd.read_csv('data/ukdale_aggreate_windowed_series_house_1.csv', header=None)

import torch
import os
from utils import get_available_device
from pyod.models.lunar import LUNAR
if torch.cuda.is_available() == True:
    device = 'cuda:0'
else:
    device = 'cpu'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.cuda.set_device(device)
clf_name = 'LUNAR'
clf = LUNAR()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# from pyod.models.rgraph import RGraph
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.auto_encoder import AutoEncoder
clf = ALAD(contamination=0.02)
clf_name = 'ALAD'
clf.fit(X_train.sample(n=2000))

import joblib
joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/ALAD_model.joblib')
clfl = ALAD(contamination=0.02)
clfl = joblib.load('F:/PycharmProjects/pythonProject/result/LODA_model.joblib')

res = clfl.predict(X_train.iloc[:1000])

from pyod.models.knn import KNN

from pyod.models.loda import LODA
clf = LODA(contamination=0.02)
clf_name = 'LODA'
clf.fit(X_train.sample(n=10000))
import joblib
joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/LODA_model.joblib')


clf = KNN(contamination=0.02)
clf_name = 'KNN'
clf.fit(X_train.sample(n=10000))
import joblib
joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/KNN_model.joblib')


from pyod.models.lscp import LSCP
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cof import COF
clf = LSCP(contamination=0.02,detector_list=[KNN(),LOF(),COF()])
clf_name = 'LSCP'
clf.fit(X_train.sample(n=10000))
import joblib
joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/LSCP_model.joblib')


from pyod.models.rod import ROD
clf = ROD(contamination=0.02)
clf_name = 'ROD'
clf.fit(X_train.sample(n=10))
import joblib
joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/ROD_model.joblib')