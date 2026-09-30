from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_JEPA_VTS_Imputation(Exp_Basic):
    """
    JEPAVTS imputation with a TimeMixer student.

    The student encodes the masked window. The task loss is MSE on the missing
    points only. The JEPA loss aligns the predictor output to the frozen DINO
    embedding of the *clean* window (the cache is hashed on the unmasked series).
    Validation and test score the masked points only; the teacher is not used.
    """

    def __init__(self, args):
        super(Exp_JEPA_VTS_Imputation, self).__init__(args)

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

    def _random_mask(self, batch_x):
        """1 = observed, 0 = missing. `inp` is the series with missing points zeroed."""
        B, T, N = batch_x.shape
        mask = torch.rand((B, T, N), device=batch_x.device)
        mask[mask <= self.args.mask_rate] = 0
        mask[mask > self.args.mask_rate] = 1
        inp = batch_x.masked_fill(mask == 0, 0)
        return inp, mask

    def _jepa_loss(self, predicted, target, loss_type='mse'):
        if loss_type == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        return torch.nn.functional.mse_loss(predicted, target)

    def _compute_jepa_loss_term(self, predicted, teacher_encodings, loss_type,
                                jepa_weight, model_ref):
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
        if isinstance(predicted, list):
            predicted = predicted[0]
        single_loss = self._jepa_loss(predicted, teacher_encodings.detach(), loss_type)
        return jepa_weight * single_loss, [single_loss]

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                inp, mask = self._random_mask(batch_x)
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                target = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs.detach()[mask == 0], target.detach()[mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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

        rendering_info = (f"Multi-rendering {model_ref.rendering_methods} "
                          f"(alpha_mode={model_ref.multi_rendering_alpha_mode})"
                          if model_ref.multi_rendering
                          else f"Single rendering {model_ref.rendering_methods}")
        print(f"\nImputation training (JEPAVTS + TimeMixer):")
        print(f"   - Rendering: {rendering_info}")
        print(f"   - Mask rate: {self.args.mask_rate}")
        print(f"   - JEPA scale: {getattr(self.args, 'timemixer_jepa_scale', 'fine')}")
        print(f"   - JEPA Weight: {jepa_weight}  | Loss: {jepa_loss_type}")
        print(f"   - Train steps/epoch: {train_steps}\n")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, train_imp_loss, train_jepa_loss = [], [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Teacher looks up the clean window. The student sees the masked one.
                teacher_encoding = model_ref.teacher_forward(batch_x)

                inp, mask = self._random_mask(batch_x)
                recon, _student_encoding, predicted_teacher_encoding = self.model(
                    inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                recon = recon[:, :, f_dim:]
                target = batch_x[:, :, f_dim:]
                point_mask = mask[:, :, f_dim:]
                imp_loss = criterion(recon[point_mask == 0], target[point_mask == 0])

                jepa_loss_term, individual_jepa_losses = self._compute_jepa_loss_term(
                    predicted_teacher_encoding, teacher_encoding,
                    jepa_loss_type, jepa_weight, model_ref)
                jepa_loss_scalar = sum(l.item() for l in individual_jepa_losses)

                loss = imp_loss + jepa_loss_term

                train_loss.append(loss.item())
                train_imp_loss.append(imp_loss.item())
                train_jepa_loss.append(jepa_loss_scalar)

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | "
                          f"loss: {loss.item():.7f} (imp: {imp_loss.item():.7f}, "
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

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} (imp: {3:.7f}, jepa: {4:.7f}) "
                  "Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                      epoch + 1, train_steps, train_loss,
                      np.average(train_imp_loss), np.average(train_jepa_loss),
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
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                inp, mask = self._random_mask(batch_x)
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    m = mask[0, :, -1].detach().cpu().numpy()
                    filled = filled * m + pred[0, :, -1] * (1 - m)
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return
