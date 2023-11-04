import pandas as pd
import numpy as np

# data_part1 = pd.read_csv('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/2023-05-01-00-00-00-p.csv')
# data_part2 = pd.read_csv('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/2023-05-08-00-00-00-p.csv')
# data_part3 = pd.read_csv('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/2023-05-15-00-00-00-p.csv')
# data_part4 = pd.read_csv('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/2023-05-22-00-00-00-p.csv')
# data_part5 = pd.read_csv('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/2023-05-29-00-00-00-p.csv')
#
# data = pd.concat([data_part1, data_part2, data_part3, data_part4, data_part5])

import glob

file_paths = glob.glob('F:/polyu work/NILM_clp/CLP/Vacation house May 2023/*.csv')
file_paths.sort()

data_parts = []
for i in range(21):  # Adjust the range according to the number of remaining files
    data_part = pd.read_csv(file_paths[i])
    data_parts.append(data_part)

data = pd.concat(data_parts)
# aggregation = data['Tai O Holiday House  - Total  (kW)']
window_size = 144
data_agg = pd.DataFrame(data['Tai O Holiday House  - Total  (kW)']).to_numpy()

def add_pattern_noise_point1(x, mag=1, ratio=0.1, window_size=144):
    abnormal = np.random.randn(x.shape[0],x.shape[1])
    max_ac = np.max(x, axis=0)
    abnormal = np.abs(abnormal * mag * max_ac)
    cutoff = np.random.rand(x.shape[0],x.shape[1])
    states = cutoff > (1-ratio/window_size)
    kk = 0
    while(kk<states.shape[0]):
        if states[kk] == True:
            states[kk:kk+window_size] = True
            kk += window_size
        else:
            kk += 1
    abnormal = abnormal * states

    return x + abnormal, states

data_noise, indi = add_pattern_noise_point1(data_agg)

data_use = pd.DataFrame(data.iloc[window_size:]['Time stamp']).to_numpy()
name = []
for j in range(1,window_size+1):
    name.append('time-' + str(j))
    data_use = np.concatenate([data_use, data_noise[(window_size-j):(-j)].reshape([-1,1])],axis=1)
    # print(data_use.shape)


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


count = []
data_n = data.fillna(0).to_numpy()
for k in range(window_size, data_n.shape[0]):
    kk = 9
    count.append(np.sum(data_n[k-window_size:k,9:20],axis=0))

count = np.array(count)
column=['time'] + name + ['主人房聽（洗衣機）4L1', '冷氣-2L1', '冷氣-2L2',
       '冷氣主人房-3L2', '冷氣分體機-2L3', '冷氣聽-3L1',
       '廚房蘇-4L3', '浴室寶-6L2', '熱水爐-5 L1',
       '熱水爐-5 L2', '熱水爐-5 L3']

data_use[:,1:], _ = add_pattern_noise(data_use[:,1:],mag=1,ratio=0.1)
# data_use = np.concatenate([data_use,count],axis=1)
data_use = np.concatenate([data_use,data_n[window_size-1:-1,9:20]],axis=1)
d_f = pd.DataFrame(columns=column,data=data_use)
d_f.to_csv('data_use_144_noise_10p.csv')

