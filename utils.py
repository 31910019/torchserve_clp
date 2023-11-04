import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

class Mydataset(Dataset):
    def __init__(self, args, data, label):
        self.label = label
        self.window_size = args.window_size
        self.x = data.to_numpy()[self.window_size:,2:self.window_size+2].astype(np.float32)
        # self.y = data[label].to_numpy().astype(np.float32)
        temp = data[label].to_numpy().astype(np.float32)
        yy = []
        for tt in range(temp.shape[0]-self.window_size):
            yy.append(temp[tt:tt+self.window_size])
        self.y = np.array(yy).reshape([temp.shape[0]-self.window_size,self.window_size])

        self.appliance_names = args.appliance_names
        self.cutoff = [args.cutoff[i]
                       for i in self.appliance_names]

        self.threshold = [args.threshold[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]

        temp_status = self.compute_status(temp)
        ss = []
        for tt in range(temp_status.shape[0]-self.window_size):
            ss.append(temp_status[tt:tt+self.window_size])
        self.status = np.array(ss).reshape([temp_status.shape[0]-self.window_size,self.window_size])

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.status[item]

    def __len__(self):
        return self.x.shape[0]

    def remove_null(self):
        for ik in range(self.x.shape[0]):
            try:
                a = np.sum(self.y[ik])
            except:
                break
            if np.sum(self.y[ik]) <= 0.01:
                self.x = np.delete(self.x, ik, axis=0)
                self.y = np.delete(self.y, ik, axis=0)

    def get_mean_std(self):
        return np.mean(self.x, axis=0), np.std(self.x, axis=0)

    def get_max(self):
        return np.max(self.x, axis=0)

    def normalize(self, x_mean=[], x_std=[]):
        if len(x_mean) == 0 or len(x_std) == 0:
            x_mean, x_std = self.get_mean_std()

        self.x = (self.x - x_mean)/x_std

    def standard(self, x_max=[]):
        if len(x_max) == 0:
            x_max = np.max(self.x, axis=0)

        self.x = self.x / x_max

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [1 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]
        # print(columns)
        # print(data.shape)
        for i in range(columns):
            # print(i)
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status

def get_user_input(args):
    import torch
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch, "has_mps", False)
    # device = "mps" if has_mps else "gpu" if has_gpu else "cpu"
    # print(f"device: {device}")
    if has_gpu:
        args.device = 'cuda:' + input('Input GPU ID: ')
    elif has_mps:
        args.device = 'mps'
    else:
        args.device = 'cpu'

    dataset_code = {'r': 'redd_lf', 'u': 'uk_dale'}
    args.dataset_code = dataset_code[input(
        'Input r for REDD, u for UK_DALE: ')]

    if args.dataset_code == 'redd_lf':
        app_dict = {
            'r': ['refrigerator'],
            'w': ['washer_dryer'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input r, w, m or d for target appliance: ')]

    elif args.dataset_code == 'uk_dale':
        app_dict = {
            'k': ['kettle'],
            'f': ['fridge'],
            'w': ['washing_machine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input k, f, w, m or d for target appliance: ')]

    args.num_epochs = int(input('Input training epochs: '))

def get_user_input_clp(args):
    import torch
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch, "has_mps", False)
    # device = "mps" if has_mps else "gpu" if has_gpu else "cpu"
    # print(f"device: {device}")
    if has_gpu:
        args.device = 'cuda:' + input('Input GPU ID: ')
    elif has_mps:
        args.device = 'mps'
    else:
        args.device = 'cpu'


    args.appliance_names = [input(
        'Input one of the below appliance with no quotation mark ("主人房聽（洗衣機）4L1","冷氣-2L1","冷氣-2L2","冷氣主人房-3L2","冷氣分體機-2L3","冷氣聽-3L1","廚房蘇-4L3","浴室寶-6L2","熱水爐-5 L1","熱水爐-5 L2","熱水爐-5 L3"): ')]
    print(args.appliance_names)

    args.num_epochs = int(input('Input training epochs: '))


def set_template(args):
    args.output_size = len(args.appliance_names)
    if args.dataset_code == 'redd_lf':
        args.window_stride = 120
        args.house_indicies = [1, 2, 3, 4, 5, 6]

        args.cutoff = {
            'aggregate': 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave': 1800,
            'dishwasher': 1200
        }

        args.threshold = {
            'refrigerator': 50,
            'washer_dryer': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'refrigerator': 10,
            'washer_dryer': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'refrigerator': 1e-6,
            'washer_dryer': 0.001,
            'microwave': 1.,
            'dishwasher': 1.
        }

    elif args.dataset_code == 'uk_dale':
        args.window_stride = 240
        args.house_indicies = [1, 2, 3, 4, 5]
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.
        }

    elif args.dataset_code == 'clp':
        # args.window_stride = 240
        # args.house_indicies = [1, 2, 3, 4, 5]

        # args.cutoff = {
        #     '主人房聽（洗衣機）4L1': 5,
        #     '冷氣-2L1': 22,
        #     '冷氣-2L2': 22,
        #     '冷氣主人房-3L2': 40,
        #     '冷氣分體機-2L3': 40,
        #     '冷氣聽-3L1': 32,
        #     '廚房蘇-4L3': 12,
        #     '浴室寶-6L2': 36,
        #     '熱水爐-5 L1': 45,
        #     '熱水爐-5 L2': 32,
        #     '熱水爐-5 L3': 24
        # }
        # args.threshold = {
        #     '主人房聽（洗衣機）4L1': 0.2,
        #     '冷氣-2L1': 1,
        #     '冷氣-2L2': 1,
        #     '冷氣主人房-3L2': 1,
        #     '冷氣分體機-2L3': 1,
        #     '冷氣聽-3L1': 1,
        #     '廚房蘇-4L3': 0.5,
        #     '浴室寶-6L2': 1,
        #     '熱水爐-5 L1': 1,
        #     '熱水爐-5 L2': 1,
        #     '熱水爐-5 L3': 1
        # }
        args.cutoff = {
            '主人房聽（洗衣機）4L1': 0.025,
            '冷氣-2L1': 1.1,
            '冷氣-2L2': 1.1,
            '冷氣主人房-3L2': 2,
            '冷氣分體機-2L3': 2,
            '冷氣聽-3L1': 1.6,
            '廚房蘇-4L3': 0.6,
            '浴室寶-6L2': 1.8,
            '熱水爐-5 L1': 2.25,
            '熱水爐-5 L2': 1.6,
            '熱水爐-5 L3': 1.2
        }

        args.threshold = {
            '主人房聽（洗衣機）4L1': 0.001,
            '冷氣-2L1': 0.05,
            '冷氣-2L2': 0.05,
            '冷氣主人房-3L2': 0.05,
            '冷氣分體機-2L3': 0.05,
            '冷氣聽-3L1': 0.05,
            '廚房蘇-4L3': 0.025,
            '浴室寶-6L2': 0.00001,
            '熱水爐-5 L1': 0.05,
            '熱水爐-5 L2': 0.05,
            '熱水爐-5 L3': 0.05
        }


        args.min_on = {
            '主人房聽（洗衣機）4L1': 0.01,
            '冷氣-2L1': 0.05,
            '冷氣-2L2': 0.05,
            '冷氣主人房-3L2': 0.05,
            '冷氣分體機-2L3': 0.05,
            '冷氣聽-3L1': 0.05,
            '廚房蘇-4L3': 0.025,
            '浴室寶-6L2': 0.05,
            '熱水爐-5 L1': 0.05,
            '熱水爐-5 L2': 0.05,
            '熱水爐-5 L3': 0.05
        }

        args.min_off = {
            '主人房聽（洗衣機）4L1': 0.01,
            '冷氣-2L1': 0.05,
            '冷氣-2L2': 0.05,
            '冷氣主人房-3L2': 0.05,
            '冷氣分體機-2L3': 0.05,
            '冷氣聽-3L1': 0.05,
            '廚房蘇-4L3': 0.025,
            '浴室寶-6L2': 0.05,
            '熱水爐-5 L1': 0.05,
            '熱水爐-5 L2': 0.05,
            '熱水爐-5 L3': 0.05
        }

        args.c0 = {
            '主人房聽（洗衣機）4L1': 0.004,
            '冷氣-2L1': 0.02,
            '冷氣-2L2': 0.02,
            '冷氣主人房-3L2': 0.02,
            '冷氣分體機-2L3': 0.02,
            '冷氣聽-3L1': 0.02,
            '廚房蘇-4L3': 0.001,
            '浴室寶-6L2': 0.02,
            '熱水爐-5 L1': 0.02,
            '熱水爐-5 L2': 0.02,
            '熱水爐-5 L3': 0.02
        }

    args.optimizer = 'adam'
    args.lr = 1e-4
    args.enable_lr_schedule = False
    args.batch_size = 128


def acc_precision_recall_f1_score(pred, status):

    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)

def acc_precision_recall_f1_score_mmoe(pred, status):
    # print('acc_precision_recall_f1_score')
    # print(pred.shape)
    # print(status.shape)
    # pred = pred[:,-1]
    assert pred.shape == status.shape

    pred[1 / (1 + np.exp(-pred)) > 0.5] = 1
    pred[1 / (1 + np.exp(-pred)) <= 0.5] = 0

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def relative_absolute_error(pred, label):

    # pred = pred[:,-1]
    # print(pred.shape)
    # print(label.shape)
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)
    # print(label[:100,i])
    # print(pred[:100,i])

    return np.array(relative), np.array(absolute)

def get_available_device():
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch, "has_mps", False)
    # device = "mps" if has_mps else "gpu" if has_gpu else "cpu"
    # print(f"device: {device}")
    if has_gpu:
        device = 'cuda' #'cuda:' + f"{int(input('Input GPU ID: ').strip())}"
    elif has_mps:
        device = 'mps'
    else:
        device = 'cpu'
    return device