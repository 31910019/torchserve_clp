import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

import os
import json
import random
import numpy as np
from abc import *
from pathlib import Path

from utils import *
import matplotlib.pyplot as plt

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.set_default_tensor_type(torch.FloatTensor)


class Trainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, stats, export_root, train_set):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.float().to(self.device)
        self.export_root = Path(export_root)

        self.cutoff = torch.tensor([args.cutoff[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.threshold = torch.tensor(
            [args.threshold[i] for i in args.appliance_names]).float().to(self.device)
        # print(self.threshold)
        self.min_on = torch.tensor([args.min_on[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.min_off = torch.tensor(
            [args.min_off[i] for i in args.appliance_names]).float().to(self.device)

        self.normalize = args.normalize
        self.denom = args.denom
        if self.normalize == 'mean':
            self.x_mean, self.x_std = stats
            self.x_mean = torch.tensor(self.x_mean).float().float().to(self.device)
            self.x_std = torch.tensor(self.x_std).float().float().to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).float().to(self.device)
        print('C0: {}'.format(self.C0))
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

        if self.args.is_od == 1:
            import pandas as pd
            X_train = pd.read_csv('data/ukdale_aggreate_windowed_series_house_1.csv', header=None)
            # m = X_train.shape[0]
            x_mean, x_std = train_set.get_mean_std()

            X_train = (X_train - x_mean)/x_std
            print(X_train.shape)
            m = X_train.shape[0]
            # X_train = X_train.to_numpy()[1:int(3*m/4),2:args.window_size+2]

            # from pyod.models.anogan import AnoGAN
            from pyod.models.alad import ALAD
            # self.clf = ALAD(contamination=0.1,latent_dim=200,enc_layers=[128,256,128],dec_layers=[128,256,128],disc_xz_layers=[128,256,128],disc_xx_layers=[128,256,128],disc_zz_layers=[128,256,128],dropout_rate=0.01,output_activation='relu',add_recon_loss=True,epochs=1000)
            self.clf = ALAD(contamination=0.025, latent_dim=2, enc_layers=[256, 256], dec_layers=[256, 256],
                            disc_xz_layers=[64, 64], disc_xx_layers=[64, 64],
                            disc_zz_layers=[64, 64], dropout_rate=0.01, output_activation='tanh',
                            add_recon_loss=True, epochs=1000)
            # print(X_train.shape)
            self.clf.fit(X_train.sample(n=2000))
            self.latent_record = self.clf.enc(X_train.astype(np.float32))
            od = self.clf.predict(X_train)
            self.latent_record = self.latent_record[od==0]

    def outlier_detection(self, datas, method, clf):
        path_od = 'F:/PycharmProjects/pythonProject/result/' + method + '_model.joblib'
        # clf = joblib.load(path_od)
        data_temp = datas

        return clf.predict(data_temp.cpu().detach().numpy())

    def outlier_calibration(self, data, od_res, clf):
        if clf.preprocessing:
            data_temp = clf.scaler_.fit_transform(data.cpu())
        else:
            data_temp = data.cpu()

        latent_curent = clf.enc(data_temp.cpu())

        res_record = []
        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                # print(latent_curent[i])
                res_record.append(data_temp[i].detach().numpy())
                distance = np.mean(np.square(np.abs(self.latent_record - latent_curent[i])), axis=1)
                # print(distance.shape)
                min_index = np.argmin(distance)
                lat_temp = self.latent_record.numpy()[min_index]
                # print(lat_temp)
                # print(type(lat_temp))
                data_temp[i] = torch.from_numpy(self.clf.dec(lat_temp.reshape([1, -1])).numpy())
                # print(data_temp[i])
                res_record.append(data_temp[i].detach().numpy())
        if len(res_record) > 0:
            # print(res_record[0].shape)
            # print(res_record[1].shape)
            at = np.load('record_draw.npy')
            at = np.concatenate([at,np.array(res_record).reshape([-1,self.args.window_size])],axis=0)
            np.save('record_draw.npy', at)

        return data_temp.to(self.device)

    def outlier_delete(self, data, od_res, clf, labels_energy):
        if clf.preprocessing:
            data_temp = clf.scaler_.fit_transform(data.cpu())
        else:
            data_temp = data.cpu()

        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                data_temp[i] = torch.zeros(1, self.args.window_size)
                labels_energy[i] = 0

        return data_temp.to(self.device), labels_energy.to(self.device)

    def train(self):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []

        best_rel_err, _, best_acc, _, _, best_f1 = self.validate()
        self._save_state_dict()

        for epoch in range(self.num_epochs):
            self.train_bert_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1 = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())

            if f1.mean() + acc.mean() - rel_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                self._save_state_dict()

    def train_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(self.device), status.float().to(self.device)
            self.optimizer.zero_grad()
            print(seqs.shape)
            logits = self.model(seqs)
            # logits_status = logits[0]
            # logits = logits[1]
            labels = labels_energy / self.cutoff
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            logits_status = self.compute_status(logits_energy)

            kl_loss = self.kl(torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits.contiguous().view(-1).float(),
                labels.contiguous().view(-1).float())
            margin_loss = self.margin((logits_status * 2 - 1).contiguous().view(-1).float(),
                (status * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss
            
            on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                    labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def train_bert_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            # print('trainshape')
            # print(labels_energy.shape)
            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(self.device), status.float().to(self.device)
            # print('seqs_shape')
            # print(seqs.shape)
            batch_shape = status.shape
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                seqs = self.outlier_calibration(seqs, od_res, self.clf)
                # clean_mo = seqs[0]
                # for i in range(seqs.shape[0]):
                #     if od_res[i] == 0:
                #         clean_mo = seqs[i]
                #         break
                # for i in range(seqs.shape[0]):
                #     if od_res[i] == 1:
                #         seqs[i] = clean_mo

            self.optimizer.zero_grad()
            logits = self.model(seqs)
            # logits_status = logits[0]
            # logits = logits[1]
            # print(logits)
            labels = labels_energy / self.cutoff
            # print(logits * self.cutoff)
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            # print('logits_energy')
            # print(logits_energy)
            logits_status = self.compute_status(logits_energy)
            
            mask = (status >= 0)
            # print('log')
            # print(logits_status.shape)
            # print(labels.shape)
            # print(mask.shape)
            labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
            logits_masked = torch.masked_select(logits.squeeze(), mask.squeeze()).view((-1, batch_shape[-1]))
            status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
            logits_status_masked = torch.masked_select(logits_status.squeeze(), mask.squeeze()).view((-1, batch_shape[-1]))

            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).float(),
                labels_masked.contiguous().view(-1).float())
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).float(),
                (status_masked * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss
            
            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                    labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                # print(seqs.shape)
                # print('valshape')
                # print(labels_energy.shape)
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(self.device), status.float().to(self.device)
                # print('seqs_shape')
                # print(seqs.shape)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

                logits = self.model(seqs)
                # logits_status = logits[0]
                # print(logits[0])
                # logits = logits[1]
                labels = labels_energy / self.cutoff
                # print(type(logits))
                # print(type(self.cutoff))
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                # print('logits_energy')
                # print(logits_energy.shape)
                # print(logits_status.shape)
                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))
                # print(average_rel_err)
                # print(average_acc)
                # print(average_f1)
                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))

        return_rel_err = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def test(self, test_loader):
        self._load_best_model()
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []

        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(self.device), status.float().to(self.device)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)
                    # clean_mo = seqs[0]
                    # for i in range(seqs.shape[0]):
                    #     if od_res[i] == 0:
                    #         clean_mo = seqs[i]
                    #         break
                    # for i in range(seqs.shape[0]):
                    #     if od_res[i] == 1:
                    #         seqs[i] = clean_mo

                logits = self.model(seqs)
                # logits_status = logits[0]
                # logits = logits[1]
                labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                # logits_no_cut = logits_energy
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                    ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                    ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                tqdm_dataloader.set_description('Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))

                label_curve.append(labels_energy.detach().cpu().numpy().tolist())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().tolist())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())

        label_curve = np.concatenate(label_curve).reshape(-1, self.args.output_size)
        e_pred_curve = np.concatenate(e_pred_curve).reshape(-1, self.args.output_size)
        status_curve = np.concatenate(status_curve).reshape(-1, self.args.output_size)
        s_pred_curve = np.concatenate(s_pred_curve).reshape(-1, self.args.output_size)

        np.save('label_curve_'+str(self.args.appliance_names), label_curve)
        np.save('e_pred_curve_'+str(self.args.appliance_names), e_pred_curve)

        # self._save_result({'gt': label_curve.tolist(),
        #     'pred': e_pred_curve.tolist()}, 'test_result.json')

        if self.args.output_size > 1:
            return_rel_err = np.array(relative_errors).mean(axis=0)
        else:
            return_rel_err = np.array(relative_errors).mean()
        return_rel_err, return_abs_err = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score(s_pred_curve, status_curve)

        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).float().to(self.device)

        # data[data < 5] = 0
        data = torch.min(data, self.cutoff.float())
        return data

    def compute_status(self, data):
        data_shape = data.shape
        columns = data.squeeze().shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).float().to(self.device)
        
        status = (data >= self.threshold) * 1
        # print(sum(sum(status)))
        return status

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.float().to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)


class Trainer_MMOE(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, stats, export_root, train_set):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.float().to(self.device)
        self.export_root = Path(export_root)

        self.cutoff = torch.tensor([args.cutoff[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.threshold = torch.tensor(
            [args.threshold[i] for i in args.appliance_names]).float().to(self.device)
        # print(self.threshold)
        self.min_on = torch.tensor([args.min_on[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.min_off = torch.tensor(
            [args.min_off[i] for i in args.appliance_names]).float().to(self.device)

        self.normalize = args.normalize
        self.denom = args.denom
        if self.normalize == 'mean':
            self.x_mean, self.x_std = stats
            self.x_mean = torch.tensor(self.x_mean).float().float().to(self.device)
            self.x_std = torch.tensor(self.x_std).float().float().to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).float().to(self.device)
        print('C0: {}'.format(self.C0))
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

        if self.args.is_od == 1:
            import pandas as pd
            # X_train = pd.read_csv('data/ukdale_aggreate_windowed_series_house_1.csv', header=None)
            # m = X_train.shape[0]
            X_train = train_set.x
            m = X_train.shape[0]
            # X_train = X_train.to_numpy()[1:int(3*m/4),2:args.window_size+2]

            # from pyod.models.anogan import AnoGAN
            from pyod.models.alad import ALAD
            # self.clf = ALAD(contamination=0.1,latent_dim=200,enc_layers=[128,256,128],dec_layers=[128,256,128],disc_xz_layers=[128,256,128],disc_xx_layers=[128,256,128],disc_zz_layers=[128,256,128],dropout_rate=0.01,output_activation='relu',add_recon_loss=True,epochs=1000)
            self.clf = ALAD(contamination=0.05)
            self.clf.fit(X_train.sample(n=2000))
            self.latent_record = self.clf.enc(X_train.astype(np.float32))
            od = self.clf.predict(X_train)
            self.latent_record = self.latent_record[od == 0]

    def outlier_detection(self, datas, method, clf):
        path_od = 'F:/PycharmProjects/pythonProject/result/' + method + '_model.joblib'
        # clf = joblib.load(path_od)
        data_temp = datas

        return clf.predict(data_temp.cpu().detach().numpy())

    def outlier_calibration(self, data, od_res, clf):
        if clf.preprocessing:
            data_temp = clf.scaler_.fit_transform(data.cpu())
        else:
            data_temp = data.cpu()

        res_record = []
        latent_curent = clf.enc(data_temp.cpu())
        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                res_record.append(latent_curent[i])
                distance = np.mean(np.square(np.abs(self.latent_record - latent_curent[i])), axis=1)
                # print(distance.shape)
                min_index = np.argmin(distance)
                lat_temp = self.latent_record.numpy()[min_index]
                res_record.append(lat_temp)
                # print(type(lat_temp))
                data_temp[i] = torch.from_numpy(self.clf.dec(lat_temp.reshape([1, -1])).numpy())
        if len(res_record) > 0:
            print(type(res_record[0]))
            print(type(res_record[1]))
            np.save('record_draw.npy', np.array(res_record).reshape([-1,self.args.window_size]))

        return data_temp.to(self.device)

    def outlier_delete(self, data, od_res, clf, labels_energy):
        if clf.preprocessing:
            data_temp = clf.scaler_.fit_transform(data.cpu())
        else:
            data_temp = data.cpu()

        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                data_temp[i] = torch.zeros(1, self.args.window_size)
                labels_energy[i] = 0

        return data_temp.to(self.device), labels_energy.to(self.device)

    def train(self):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []

        best_rel_err, _, best_acc, _, _, best_f1 = self.validate()
        self._save_state_dict()

        for epoch in range(self.num_epochs):
            self.train_bert_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1 = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())

            if f1.mean() + acc.mean() - rel_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                self._save_state_dict()

    def train_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                self.device), status.float().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(seqs)
            # logits_status = logits[0]
            # logits = logits[1]
            labels = labels_energy / self.cutoff
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            logits_status = self.compute_status(logits_energy)

            kl_loss = self.kl(torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9),
                              F.softmax(labels.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits.contiguous().view(-1).float(),
                                labels.contiguous().view(-1).float())

            margin_loss = self.margin((logits_status * 2 - 1).contiguous().view(-1).float(),
                                      (status * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss

            on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                                        labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size

            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def train_bert_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            # print('trainshape')
            # print(labels_energy.shape)
            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                self.device), status.float().to(self.device)
            # print('seqs_shape')
            # print(seqs.shape)
            batch_shape = status.shape
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

            self.optimizer.zero_grad()
            logits = self.model(seqs)
            logits_status = logits[0]
            logits = logits[1]
            # print(logits)
            labels = labels_energy / self.cutoff
            # print(logits * self.cutoff)
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            # print('logits_energy')
            # print(logits_energy)
            # logits_status = self.compute_status(logits_energy)

            mask = (status >= 0)
            labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
            logits_masked = torch.masked_select(logits.squeeze(), mask.squeeze()).view((-1, batch_shape[-1]))
            status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
            logits_status_masked = torch.masked_select(logits_status.squeeze(), mask.squeeze()).view(
                (-1, batch_shape[-1]))

            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9),
                              F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).float(),
                                labels_masked.contiguous().view(-1).float())
            # print('start')
            # print(logits_masked)
            # print(labels_masked)
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).float(),
                                      (status_masked * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss

            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                                        labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size

            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                # print(seqs.shape)
                # print('valshape')
                # print(labels_energy.shape)
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                    self.device), status.float().to(self.device)
                # print('seqs_shape')
                # print(seqs.shape)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

                logits = self.model(seqs)
                logits_status = logits[0]
                # print(logits[0])
                logits = logits[1]
                labels = labels_energy / self.cutoff

                logits_energy = self.cutoff_energy(logits * self.cutoff)
                comp_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * comp_status
                # logits_energy[logits_energy < 0.01] = 0

                # print(logits_energy)
                # print(labels_energy)

                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score_mmoe(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))

        return_rel_err = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def test(self, test_loader):
        self._load_best_model()
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                    self.device), status.float().to(self.device)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

                logits = self.model(seqs)
                logits_status = logits[0]
                logits = logits[1]
                labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                comp_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * comp_status
                # logits_energy[logits_energy < 0.01] = 0

                acc, precision, recall, f1 = acc_precision_recall_f1_score_mmoe(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                tqdm_dataloader.set_description('Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))

                label_curve.append(labels_energy.detach().cpu().numpy().tolist())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().tolist())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())

        label_curve = np.concatenate(label_curve).reshape(-1, self.args.output_size)
        e_pred_curve = np.concatenate(e_pred_curve).reshape(-1, self.args.output_size)
        status_curve = np.concatenate(status_curve).reshape(-1, self.args.output_size)
        s_pred_curve = np.concatenate(s_pred_curve).reshape(-1, self.args.output_size)

        self._save_result({'gt': label_curve.tolist(),
                           'pred': e_pred_curve.tolist()}, 'test_result.json')

        if self.args.output_size > 1:
            return_rel_err = np.array(relative_errors).mean(axis=0)
        else:
            return_rel_err = np.array(relative_errors).mean()
        return_rel_err, return_abs_err = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score_mmoe(s_pred_curve,
                                                                                               status_curve)

        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).float().to(self.device)

        # data[data < 5] = 0
        data = torch.min(data, self.cutoff.float())
        return data

    def compute_status(self, data):
        data_shape = data.shape
        columns = data.squeeze().shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).float().to(self.device)

        status = (data >= self.threshold) * 1
        # print(sum(sum(status)))
        return status

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.float().to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)


class Trainer_clp(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, stats, export_root, train_set):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.float().to(self.device)
        self.export_root = Path(export_root)

        self.cutoff = torch.tensor([args.cutoff[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.threshold = torch.tensor(
            [args.threshold[i] for i in args.appliance_names]).float().to(self.device)
        # print(self.threshold)
        self.min_on = torch.tensor([args.min_on[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.min_off = torch.tensor(
            [args.min_off[i] for i in args.appliance_names]).float().to(self.device)

        self.normalize = args.normalize
        self.denom = args.denom
        if self.normalize == 'mean':
            self.x_mean, self.x_std = stats
            self.x_mean = torch.tensor(self.x_mean).float().float().to(self.device)
            self.x_std = torch.tensor(self.x_std).float().float().to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.res_record = []
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).float().to(self.device)
        print('C0: {}'.format(self.C0))
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

        if self.args.is_od == 1:
            import pandas as pd
            # X_train = pd.read_csv('data_use_144_noise.csv', header=None)
            X_train = train_set.x
            m = X_train.shape[0]
            # X_train = X_train[:, 2:args.window_size + 2]

            if self.args.method == 'ALAD':
                # from pyod.models.anogan import AnoGAN
                from pyod.models.alad import ALAD

                # self.clf = ALAD(contamination=0.02, latent_dim=200, enc_layers=[128, 256, 128], dec_layers=[128, 256, 128],
                #                 disc_xz_layers=[128, 256, 128], disc_xx_layers=[128, 256, 128],
                #                 disc_zz_layers=[128, 256, 128], dropout_rate=0.01, output_activation='relu',
                #                 add_recon_loss=True, epochs=1000)
                self.clf = ALAD(contamination=0.05)
                self.clf.fit(X_train)
                self.latent_record = self.clf.enc(X_train.astype(np.float32))
                od = self.clf.predict(X_train)
                self.latent_record = self.latent_record[od==0]

            elif self.args.method == 'KNN':
                from pyod.models.iforest import IForest
                from pyod.models.knn import KNN
                self.clf = IForest(contamination=0.025)
                clf_name = 'KNN'
                self.clf.fit(X_train)

    def outlier_detection(self, datas, method, clf):
        path_od = 'F:/PycharmProjects/pythonProject/result/' + method + '_model.joblib'
        # clf = joblib.load(path_od)
        data_temp = datas

        return clf.predict(data_temp.cpu().detach().numpy())

    def outlier_calibration(self, data, od_res, clf):
        if clf.preprocessing:
            data_temp = clf.scaler_.fit_transform(data.cpu())
        else:
            data_temp = data.cpu()


        res_record = []
        latent_curent = clf.enc(data_temp.cpu())
        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                # print(i)
                res_record.append(latent_curent[i])
                distance = np.mean(np.square(np.abs(self.latent_record - latent_curent[i])), axis=1)
                # print(distance.shape)
                min_index = np.argmin(distance)
                # print(self.latent_record.numpy().shape)
                lat_temp = self.latent_record.numpy()[min_index]
                # print(self.res_record[-1])
                res_record.append(lat_temp)
                ttr = np.concatenate([data_temp[i].detach().numpy().reshape([1,-1]),self.clf.dec(lat_temp.reshape([1, -1])).numpy()],axis=1)
                self.res_record.append(ttr)

                # print(self.res_record[-1])
                data_temp[i] = torch.from_numpy(self.clf.dec(lat_temp.reshape([1, -1])).numpy())
        if len(res_record) > 0:
            print(type(res_record[0]))
            print(type(res_record[1]))
            np.save('record_draw.npy', np.array(res_record).reshape([-1,self.args.window_size]))

        return data_temp.to(self.device)

    def outlier_delete(self, data, od_res, clf, labels_energy):
        # if clf.preprocessing:
        #     data_temp = clf.scaler_.fit_transform(data.cpu())
        # else:
        data_temp = data.cpu()

        for i in range(data_temp.shape[0]):
            if od_res[i] == 1:
                data_temp[i] = torch.zeros(1, self.args.window_size)
                labels_energy[i] = 0

        return data_temp.to(self.device), labels_energy.to(self.device)
    def train(self):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []

        best_rel_err, _, best_acc, _, _, best_f1 = self.validate()
        self._save_state_dict()

        for epoch in range(self.num_epochs):
            self.train_bert_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1 = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())

            if f1.mean() + acc.mean() - rel_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                self._save_state_dict()

        example_input = torch.randn([self.args.batch_size, self.args.window_size])
        example_input = example_input.float()
        print(example_input.dtype)
        traced_script_module = torch.jit.trace(self.model, example_input.to(self.args.device))
        traced_script_module.save("transformer.pt")

    def train_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)

            # print(seqs.shape)
            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                self.device), status.float().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(seqs)
            # logits_status = logits[0]
            # logits = logits[1]
            labels = labels_energy / self.cutoff
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            logits_status = self.compute_status(logits_energy)

            kl_loss = self.kl(torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9),
                              F.softmax(labels.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits.contiguous().view(-1).float(),
                                labels.contiguous().view(-1).float())
            margin_loss = self.margin((logits_status * 2 - 1).contiguous().view(-1).float(),
                                      (status * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss

            on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                                        labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size

            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def train_bert_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                self.device), status.float().to(self.device)
            # print('seqs_shape')
            # print(seqs.shape)
            batch_shape = status.shape
            if self.args.is_od:
                od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                # seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)
                seqs = self.outlier_calibration(seqs, od_res, self.clf)
                # print(sum(od_res))

            self.optimizer.zero_grad()
            logits = self.model(seqs)
            # logits_status = logits[0]
            # logits = logits[1]
            # print(logits)
            labels = labels_energy / self.cutoff
            # print(logits * self.cutoff)
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            # print('logits_energy')
            # print(logits_energy)
            logits_status = self.compute_status(logits_energy)

            mask = (status >= 0)
            # print('log')
            # print(logits_status.shape)
            # print(labels.shape)
            # print(mask.shape)
            # print(logits.squeeze().shape)
            labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
            logits_masked = torch.masked_select(logits.squeeze(), mask.squeeze()).view((-1, batch_shape[-1]))
            status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
            logits_status_masked = torch.masked_select(logits_status.squeeze(), mask).view((-1, batch_shape[-1]))

            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9),
                              F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).float(),
                                labels_masked.contiguous().view(-1).float())
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).float(),
                                      (status_masked * 2 - 1).contiguous().view(-1).float())
            total_loss = kl_loss + mse_loss + margin_loss

            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                                        labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size

            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                # print(seqs.shape)
                # print('valshape')
                # print(labels_energy.shape)
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                    self.device), status.float().to(self.device)
                # print('seqs_shape')
                # print(seqs.shape)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)
                    # print(sum(od_res))

                logits = self.model(seqs)
                # logits_status = logits[0]
                # logits = logits[1]
                labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                # print('logits_energy')
                # print(logits_energy.shape)
                # print(logits_status.shape)
                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))
                # print(average_rel_err)
                # print(average_acc)
                # print(average_f1)
                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))
                # print('x')
                # print(logits_energy.shape)
                # print(labels_energy.shape)
        return_rel_err = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)
        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def test(self, test_loader):
        self._load_best_model()
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values, = [], [], [], []

        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.float().to(self.device), labels_energy.float().to(
                    self.device), status.float().to(self.device)
                if self.args.is_od:
                    od_res = self.outlier_detection(seqs, self.args.method, self.clf)
                    seqs, labels_energy = self.outlier_delete(seqs, od_res, self.clf, labels_energy)
                    # clean_mo = seqs[0]
                    # for i in range(seqs.shape[0]):
                    #     if od_res[i] == 0:
                    #         clean_mo = seqs[i]
                    #         break
                    # for i in range(seqs.shape[0]):
                    #     if od_res[i] == 1:
                    #         seqs[i] = clean_mo

                logits = self.model(seqs)
                # logits_status = logits[0]
                # logits = logits[1]
                labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                tqdm_dataloader.set_description('Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))

                label_curve.append(labels_energy.detach().cpu().numpy())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().squeeze())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())

        label_curve_save = np.concatenate(label_curve)[:,0]
        e_pred_curve_save = np.concatenate(e_pred_curve)[:,0]
        label_curve = np.concatenate(label_curve).reshape(-1, self.args.output_size)
        e_pred_curve = np.concatenate(e_pred_curve).reshape(-1, self.args.output_size)

        status_curve = np.concatenate(status_curve).reshape(-1, self.args.output_size)
        s_pred_curve = np.concatenate(s_pred_curve).reshape(-1, self.args.output_size)

        self._save_result({'gt': label_curve.tolist(),
                           'pred': e_pred_curve.tolist()}, 'test_result.json')

        if self.args.output_size > 1:
            return_rel_err = np.array(relative_errors).mean(axis=0)
        else:
            return_rel_err = np.array(relative_errors).mean()
        return_rel_err, return_abs_err = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score(s_pred_curve,
                                                                                               status_curve)

        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).float().to(self.device)

        # data[data < 5] = 0
        data = torch.min(data, self.cutoff.float())
        return data

    def compute_status(self, data):
        data_shape = data.shape
        columns = data.squeeze().shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).float().to(self.device)

        status = (data >= self.threshold) * 1
        # print(sum(sum(status)))
        return status

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.float().to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)


class Trainer_CNN(Trainer):
    def __init__(self, args, model, train_loader, val_loader, stats, export_root):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.float().to(self.device)
        self.export_root = Path(export_root)

        self.cutoff = torch.tensor([args.cutoff[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.threshold = torch.tensor(
            [args.threshold[i] for i in args.appliance_names]).float().to(self.device)
        print(self.threshold)
        self.min_on = torch.tensor([args.min_on[i]
                                    for i in args.appliance_names]).float().to(self.device)
        self.min_off = torch.tensor(
            [args.min_off[i] for i in args.appliance_names]).float().to(self.device)

        self.normalize = args.normalize
        self.denom = args.denom
        if self.normalize == 'mean':
            self.x_mean, self.x_std = stats
            self.x_mean = torch.tensor(self.x_mean).float().float().to(self.device)
            self.x_std = torch.tensor(self.x_std).float().float().to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).float().to(self.device)
        print('C0: {}'.format(self.C0))
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

        if self.args.is_od == 1:
            import pandas as pd
            X_train = pd.read_csv('data_use_144.csv', header=None)
            m = X_train.shape[0]
            X_train = X_train.to_numpy()[1:int(3*m/4),2:args.window_size+2]

            # from pyod.models.anogan import AnoGAN
            from pyod.models.alad import ALAD
            # self.clf = ALAD(contamination=0.1,latent_dim=200,enc_layers=[128,256,128],dec_layers=[128,256,128],disc_xz_layers=[128,256,128],disc_xx_layers=[128,256,128],disc_zz_layers=[128,256,128],dropout_rate=0.01,output_activation='relu',add_recon_loss=True,epochs=1000)
            self.clf = ALAD(contamination=0.025)
            self.clf.fit(X_train)
            self.latent_record = self.clf.enc(X_train.astype(np.float32))
            od = self.clf.predict(X_train)
            self.latent_record = self.latent_record[od==0]


    def train(self):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []

        best_rel_err, _, best_acc, _, _, best_f1 = self.validate()
        self._save_state_dict()

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1 = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())

            if f1.mean() + acc.mean() - rel_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                self._save_state_dict()