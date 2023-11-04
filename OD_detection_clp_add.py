from __future__ import division
from __future__ import print_function

import os
import sys

from pyod.models.lunar import LUNAR
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

import numpy as np
from pathlib import Path
import pandas as pd
import os
from utils import *
# "主人房聽（洗衣機）4L1","冷氣-2L1","冷氣-2L2","冷氣主人房-3L2","冷氣分體機-2L3","冷氣聽-3L1","廚房蘇-4L3","浴室寶-6L2","熱水爐-5 L1","熱水爐-5 L2","熱水爐-5 L3")

data = pd.read_csv('data_use_144.csv')
m, T = data.shape
split = int(3 * m / 4)
# train_set = Mydataset(args, data[:split], label)
label = '冷氣-2L1'
window_size = 144
x = data.to_numpy()[window_size:, 2:window_size + 2].astype(np.float32)
# self.y = data[label].to_numpy().astype(np.float32)
temp = data[label].to_numpy().astype(np.float32)
yy = []
for tt in range(temp.shape[0] - window_size):
    yy.append(temp[tt:tt + window_size])
y = np.array(yy).reshape([temp.shape[0] - window_size, window_size])

# add noise
def add_point_noise(x, mag=1, ratio=0.025):
    abnormal = np.random.randn(x.shape[0],x.shape[1])
    max_ac = np.max(x, axis=0)
    abnormal = np.abs(abnormal * mag * max_ac)
    cutoff = np.random.rand(x.shape[0],x.shape[1])
    states = cutoff > (1-ratio)
    abnormal = abnormal * states
    # indicator = np.sum(states,axis=1) > 0

    return x + abnormal, states

def add_pattern_noise(x, mag=1, ratio=0.025):
    abnormal = np.random.randn(x.shape[0], x.shape[1])
    max_ac = np.max(x, axis=0)
    abnormal = np.abs(abnormal * mag * max_ac)
    cutoff = np.random.rand(x.shape[0])
    states = cutoff > (1 - ratio)
    abnormal = (abnormal.transpose() * states.transpose()).transpose()

    return x + abnormal, states


# x, indi_point = add_point_noise(x, 1)
# x, indi_pattern = add_pattern_noise(x,100)

X_train = x[:split]
X_test = x[split:]
y_train = y[:split]
y_test = y[split:]

X_train, indi_point_train = add_point_noise(X_train, 1)
X_train, indi_pattern_train = add_pattern_noise(X_train, 1)
X_test, indi_point_test = add_point_noise(X_test,1)
X_test, indi_pattern_test = add_pattern_noise(X_test, 1)

# try:
#     indi_point_train = indi_point[:split]
#     indi_point_test = indi_point[split:]
# except:
#     pass
#
# try:
#     indi_pattern_train = indi_pattern[:split]
#     indi_pattern_test = indi_pattern[split:]
# except:
#     pass
#
from pyod.models.alad import ALAD
from pyod.models.knn import KNN
# clf = ALAD(contamination=0.025)
# clf = ALAD(contamination=0.025)
clf = ALAD(contamination=0.025, latent_dim=2, enc_layers=[256, 256], dec_layers=[256, 256],
                            disc_xz_layers=[64, 64], disc_xx_layers=[64, 64],
                            disc_zz_layers=[64, 64], dropout_rate=0.01, output_activation='tanh',
                            add_recon_loss=True, epochs=1000)
clf_name = 'ALAD'
clf.fit(X_train)

lat = clf.enc(X_train)

import matplotlib.pyplot as plt

plt.scatter(lat[:500,0], lat[:500,1])


n = 500
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.25, min_samples=10)
res = clustering.fit_predict(lat[:n])
print(max(res))

plt.close()
plt.scatter(lat[:n,0][res==0], lat[:n,1][res==0],c='b')
plt.scatter(lat[:n,0][res==1], lat[:n,1][res==1],c='r')
plt.scatter(lat[:n,0][res==2], lat[:n,1][res==2],c='g')
plt.scatter(lat[:n,0][res==3], lat[:n,1][res==3],c='c')
plt.scatter(lat[:n,0][res==4], lat[:n,1][res==4],c='m')
plt.scatter(lat[:n,0][res==-1], lat[:n,1][res==-1],c='y',label='anomaly')
plt.legend(loc='upper right')
plt.savefig('representation.pdf')
plt.show()



# import joblib
# joblib.dump(clf, 'F:/PycharmProjects/pythonProject/result/ALAD_model.joblib')
# # clfl = ALAD(contamination=0.02)
# # clfl = joblib.load('F:/PycharmProjects/pythonProject/result/LODA_model.joblib')
#
# res = clf.predict(X_test)
#
#
# # det = sum(res[np.where(indi_pattern_test>0)])
# # acc = det/res[np.where(indi_pattern_test>0)].shape[0]
#
#
#
# from pyod.models.knn import KNN
# clf_2 = KNN(contamination=0.025)
# clf_name = 'ALAD'
# clf_2.fit(X_train[:,0].reshape([-1,1]))
# res_2 = clf_2.predict(X_test[:,0].reshape([-1,1]))
#
# # indi_point_test = indi_point_test[:,0]
# det = sum(res_2[np.where(indi_point_test>0)])
# acc = det/res_2[np.where(indi_point_test>0)].shape[0]
