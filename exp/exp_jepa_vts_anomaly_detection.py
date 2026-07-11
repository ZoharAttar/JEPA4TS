from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_JEPA_VTS_Anomaly_Detection(Exp_Basic):
    """
    JEPAVTS (VibeTS) reconstruction-based anomaly detection.

    Training loss = reconstruction MSE + jepa_weight * alignment loss, where the
    alignment loss pulls the student encoding of each window toward the frozen
    per-variable DINO teacher embedding of that same window (loaded from cache).
    The teacher is queried ONLY during training; validation/test scoring uses the
    reconstruction error alone (identical to the plain anomaly baseline).
    """

    def __init__(self, args):
        super(Exp_JEPA_VTS_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        from models.JEPAVTS import Model as JEPAVTS
        model = JEPAVTS(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _jepa_loss(self, predicted, target, loss_type='mse'):
        """Alignment loss between predicted and a single target encoding."""
        if loss_type == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        return torch.nn.functional.mse_loss(predicted, target)

    def _compute_jepa_loss_term(self, predicted, teacher_encodings, loss_type,
                                jepa_weight, model_ref):
        """Total alpha-weighted alignment loss (single- or multi-rendering)."""
        if isinstance(teacher_encodings, list):
            k = len(teacher_encodings)
            if isinstance(predicted, list):
                assert len(predicted) == k
                individual_losses = [
                    self._jepa_loss(p, t.detach(), loss_type)
                    for p, t in zip(predicted, teacher_encodings)
                ]
            else:
                individual_losses = [
                    self._jepa_loss(predicted, t.detach(), loss_type)
                    for t in teacher_encodings
                ]
            alpha_mode = model_ref.multi_rendering_alpha_mode
            if alpha_mode == 'divided':
                term = (jepa_weight / k) * sum(individual_losses)
            elif alpha_mode == 'per_method':
                per_alphas = model_ref.per_method_alphas
                if per_alphas is None or len(per_alphas) != k:
                    raise ValueError(f"per_method_alphas must have {k} values, got {per_alphas}")
                term = sum(a * l for a, l in zip(per_alphas, individual_losses))
            else:
                term = jepa_weight * sum(individual_losses)
            return term, individual_losses
        else:
            if isinstance(predicted, list):
                predicted = predicted[0]
            single_loss = self._jepa_loss(predicted, teacher_encodings.detach(), loss_type)
            return jepa_weight * single_loss, [single_loss]

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs.detach(), batch_x.detach())
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        jepa_weight = getattr(self.args, 'jepa_weight', 1.0)
        jepa_loss_type = getattr(self.args, 'jepa_loss_type', 'mse')
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        print(f"\n🎯 Anomaly-detection training (VibeTS):")
        print(f"   - Architecture: {'DUAL' if model_ref.use_dual_encoder else 'SINGLE'} ENCODER")
        print(f"   - JEPA Weight: {jepa_weight}  | Loss: {jepa_loss_type}")
        print(f"   - Train steps/epoch: {train_steps}\n")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, train_rec_loss, train_jepa_loss = [], [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # Teacher (frozen, cached, no grad) — same window the student sees.
                teacher_encoding = model_ref.teacher_forward(batch_x)

                model_outputs = self.model(batch_x, None, None, None)
                if model_ref.use_dual_encoder:
                    outputs = model_outputs[0]
                    predicted_teacher_encoding = model_outputs[1]
                else:
                    outputs = model_outputs[0]
                    predicted_teacher_encoding = model_outputs[2]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                rec_loss = criterion(outputs, batch_x)

                jepa_loss_term, individual_jepa_losses = self._compute_jepa_loss_term(
                    predicted_teacher_encoding, teacher_encoding,
                    jepa_loss_type, jepa_weight, model_ref)
                jepa_loss_scalar = sum(l.item() for l in individual_jepa_losses)

                loss = rec_loss + jepa_loss_term

                train_loss.append(loss.item())
                train_rec_loss.append(rec_loss.item())
                train_jepa_loss.append(jepa_loss_scalar)

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | "
                          f"loss: {loss.item():.7f} (rec: {rec_loss.item():.7f}, "
                          f"jepa: {jepa_loss_scalar:.7f})")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} (rec: {3:.7f}, jepa: {4:.7f}) "
                  "Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                      epoch + 1, train_steps, train_loss,
                      np.average(train_rec_loss), np.average(train_jepa_loss),
                      vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) statistics on the train set (reconstruction error only; no teacher)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                attens_energy.append(score.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                attens_energy.append(score.detach().cpu().numpy())
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        gt = test_labels.astype(int)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n\n')
        f.close()
        return
