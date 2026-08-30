from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import csv
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_JEPA_VTS_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_JEPA_VTS_Long_Term_Forecast, self).__init__(args)
    
    def _build_model(self):
        # Import JEPA-VTS model
        from models.JEPAVTS import Model as JEPAVTS
        model = JEPAVTS(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        # Only student and predictor are trainable (teacher is frozen)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        return nn.MSELoss()
    
    def _jepa_loss(self, predicted, target, loss_type='mse'):
        """JEPA alignment loss between predicted and a single target encoding."""
        if loss_type == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        else:
            return torch.nn.functional.mse_loss(predicted, target)
    
    def _compute_jepa_loss_term(self, predicted, teacher_encodings, loss_type,
                                jepa_weight, model_ref):
        """
        Compute the total JEPA loss term (already multiplied by alpha).
        
        Supports three modes:
          1. Single rendering, single predictor: predicted=tensor, teachers=tensor
          2. Multi-rendering, shared predictor:   predicted=tensor, teachers=list
          3. Multi-rendering, multi-predictor:    predicted=list,   teachers=list
             Each predictor is paired with its rendering: L2(z'_i, z_i)
        
        Returns:
            jepa_loss_term: scalar loss (alpha-weighted sum of individual losses)
            individual_losses: list of per-rendering losses (for logging)
        """
        if isinstance(teacher_encodings, list):
            # Multi-rendering mode
            k = len(teacher_encodings)
            
            if isinstance(predicted, list):
                # Multi-predictor: pair each z'_i with z_i
                assert len(predicted) == k, (
                    f"Number of predictors ({len(predicted)}) must match "
                    f"number of renderings ({k})")
                individual_losses = [
                    self._jepa_loss(pred_i, te_i.detach(), loss_type)
                    for pred_i, te_i in zip(predicted, teacher_encodings)
                ]
            else:
                # Shared predictor: same z' compared to every z_i
                individual_losses = [
                    self._jepa_loss(predicted, te.detach(), loss_type)
                    for te in teacher_encodings
                ]
            
            alpha_mode = model_ref.multi_rendering_alpha_mode
            
            if alpha_mode == 'divided':
                jepa_loss_term = (jepa_weight / k) * sum(individual_losses)
            elif alpha_mode == 'per_method':
                per_alphas = model_ref.per_method_alphas
                if per_alphas is None or len(per_alphas) != k:
                    raise ValueError(
                        f"per_method_alphas must have {k} values, got {per_alphas}")
                jepa_loss_term = sum(
                    a * l for a, l in zip(per_alphas, individual_losses))
            else:
                # 'same' (default)
                jepa_loss_term = jepa_weight * sum(individual_losses)
            
            return jepa_loss_term, individual_losses
        else:
            # Single rendering mode (backward compatible)
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
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Inference mode (only returns predictions)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                loss = criterion(outputs, batch_y)
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
        
        csv_path = os.path.join(path, 'training_log.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'train_pred_loss', 'train_jepa_loss',
                             'train_horizon_loss', 'vali_loss', 'test_loss'])
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        jepa_weight = getattr(self.args, 'jepa_weight', 1.0)
        jepa_loss_type = getattr(self.args, 'jepa_loss_type', 'mse')
        horizon_weight = getattr(self.args, 'horizon_weight', 1.0)
        
        # Get model reference for architecture check
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        if getattr(model_ref, 'no_dino', False):
            arch_type = 'NO DINO (dual, no teacher/JEPA)'
        elif getattr(model_ref, 'dino_direct', False):
            arch_type = 'DINO DIRECT (single + DINO into head)'
        else:
            arch_type = 'DUAL ENCODER' if model_ref.use_dual_encoder else 'SINGLE ENCODER'
        rendering_info = (f"Multi-rendering {model_ref.rendering_methods} "
                          f"(alpha_mode={model_ref.multi_rendering_alpha_mode})"
                          if model_ref.multi_rendering
                          else "Single rendering")
        
        print(f"\nTraining Configuration:")
        print(f"   - Architecture: {arch_type}")
        print(f"   - Rendering: {rendering_info}")
        print(f"   - JEPA Weight: {jepa_weight}")
        print(f"   - JEPA Loss Type: {jepa_loss_type}")
        if getattr(model_ref, 'masked_horizon', False):
            print(f"   - Masked-horizon: on (weight={horizon_weight})")
        print(f"   - Batch Size: {self.args.batch_size}")
        print(f"   - Learning Rate: {self.args.learning_rate}")
        print(f"   - Train Steps per Epoch: {train_steps}\n")
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_pred_loss = []
            train_jepa_loss = []
            train_horizon_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 1. Teacher forward (frozen, no grad). Skipped for NO-DINO / DINO-DIRECT.
                no_jepa = getattr(model_ref, 'no_dino', False) or getattr(model_ref, 'dino_direct', False)
                if no_jepa:
                    teacher_encoding = None
                else:
                    teacher_encoding = model_ref.teacher_forward(batch_x)
                
                # 2. Student forward - handle all architectures
                model_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                predicted_horizon = None
                if getattr(model_ref, 'masked_horizon', False):
                    predicted_horizon = model_outputs[-1]
                    model_outputs = model_outputs[:-1]

                if no_jepa:
                    # NO-DINO / DINO-DIRECT: (predictions, None) - task loss only.
                    predictions = model_outputs[0]
                    predicted_teacher_encoding = None
                elif model_ref.use_multi_encoder:
                    # Multi-encoder: (predictions, student_encodings_list, predicted_list)
                    predictions = model_outputs[0]
                    predicted_teacher_encoding = model_outputs[2]
                elif model_ref.use_dual_encoder:
                    # Dual encoder: (predictions, predicted_teacher_encoding)
                    predictions = model_outputs[0]
                    predicted_teacher_encoding = model_outputs[1]
                else:
                    # Single encoder: (predictions, student_encoding, predicted_teacher_encoding)
                    predictions = model_outputs[0]
                    student_encoding = model_outputs[1]
                    predicted_teacher_encoding = model_outputs[2]
                
                # 3. Prediction loss
                f_dim = -1 if self.args.features == 'MS' else 0
                pred_outputs = predictions[:, -self.args.pred_len:, f_dim:]
                true_outputs = batch_y[:, -self.args.pred_len:, f_dim:]
                pred_loss = criterion(pred_outputs, true_outputs)
                
                # 4. JEPA alignment loss (handles both single & multi-rendering).
                #    NO-DINO: no teacher/predictor -> skip JEPA entirely.
                if predicted_teacher_encoding is None:
                    jepa_loss_term = 0.0
                    individual_jepa_losses = []
                    jepa_loss_scalar = 0.0
                else:
                    jepa_loss_term, individual_jepa_losses = self._compute_jepa_loss_term(
                        predicted_teacher_encoding, teacher_encoding,
                        jepa_loss_type, jepa_weight, model_ref)
                    jepa_loss_scalar = sum(l.item() for l in individual_jepa_losses)
                
                # 5. Combined loss
                if model_ref.use_learned_loss_weights and predicted_teacher_encoding is not None:
                    # For learned weights, pass the raw sum (unweighted) of JEPA losses
                    raw_jepa = sum(individual_jepa_losses)
                    loss, weight_info = model_ref.compute_weighted_loss(pred_loss, raw_jepa)
                else:
                    loss = pred_loss + jepa_loss_term
                    weight_info = None

                horizon_loss_scalar = 0.0
                if predicted_horizon is not None:
                    horizon_target = model_ref.horizon_teacher_forward(batch_x)
                    horizon_raw = self._jepa_loss(
                        predicted_horizon, horizon_target.detach(), jepa_loss_type)
                    loss = loss + horizon_weight * horizon_raw
                    horizon_loss_scalar = horizon_raw.item()

                train_loss.append(loss.item())
                train_pred_loss.append(pred_loss.item())
                train_jepa_loss.append(jepa_loss_scalar)
                train_horizon_loss.append(horizon_loss_scalar)
                
                if (i + 1) % 100 == 0:
                    print(f"\tIter: {i+1}/{train_steps}, Epoch: {epoch+1}/{self.args.train_epochs} [{arch_type}]")
                    print(f"\t   Total Loss: {loss.item():.7f}")
                    print(f"\t   Pred Loss: {pred_loss.item():.7f}")
                    if model_ref.multi_rendering:
                        for idx, (method, jl) in enumerate(
                                zip(model_ref.rendering_methods, individual_jepa_losses)):
                            print(f"\t   JEPA Loss [{method}]: {jl.item():.7f}")
                        print(f"\t   JEPA Loss (combined term): {jepa_loss_term.item():.7f}")
                    else:
                        print(f"\t   JEPA Loss: {jepa_loss_scalar:.7f}")
                    if predicted_horizon is not None:
                        print(f"\t   Horizon Loss: {horizon_loss_scalar:.7f}")
                    if weight_info:
                        print(f"\t   Learned Weights: pred={weight_info['pred_weight']:.4f}, jepa={weight_info['jepa_weight']:.4f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\t    Speed: {speed:.4f}s/iter; Left: {left_time/60:.2f}min\n')
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print(f"Epoch {epoch+1} completed in {(time.time() - epoch_time)/60:.2f} minutes")
            train_loss = np.average(train_loss)
            train_pred_loss = np.average(train_pred_loss)
            train_jepa_loss = np.average(train_jepa_loss)
            train_horizon_loss = np.average(train_horizon_loss)
            
            print(f"Validating...")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.7f} (Pred: {train_pred_loss:.7f}, "
                  f"JEPA: {train_jepa_loss:.7f}, Horizon: {train_horizon_loss:.7f})")
            print(f"   Vali Loss:  {vali_loss:.7f}")
            print(f"   Test Loss:  {test_loss:.7f}")
            print(f"{'='*70}\n")
            
            csv_writer.writerow([epoch + 1, train_loss, train_pred_loss, train_jepa_loss,
                                 train_horizon_loss, vali_loss, test_loss])
            csv_file.flush()
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        csv_file.close()
        print(f"Training log saved to {csv_path}")
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded from {best_model_path}")
        
        return self.model
    
    def _load_transfer_state_dict(self, state_dict):
        """Load a source checkpoint tolerating shape-mismatched tensors.

        Keeps only tensors whose shape matches the current (target) model; keys that
        differ (e.g. the TimeMixer student's enc_in-sized RevIN affine when the target
        has a different number of variables) are skipped and keep their init. Uses
        strict=False so skipped keys are simply left as 'missing'.
        """
        model_sd = self.model.state_dict()
        filtered, dropped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                dropped.append(k)
        self.model.load_state_dict(filtered, strict=False)
        if dropped:
            preview = ', '.join(dropped[:6]) + ('...' if len(dropped) > 6 else '')
            print('[transfer] skipped {} shape-mismatched/unknown tensor(s): {}'.format(len(dropped), preview))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            transfer_ckpt = getattr(self.args, 'transfer_checkpoint', '')
            ckpt_path = transfer_ckpt if transfer_ckpt else os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            print('Loading model from {}...'.format(ckpt_path))
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if transfer_ckpt:
                # Cross-dataset transfer: skip checkpoint tensors whose shape doesn't
                # match the target model (e.g. TimeMixer RevIN affines of shape [enc_in]).
                self._load_transfer_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        noise_ctx = self._make_noise_ctx(test_data)

        self.model.eval()
        print("Running test...")
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x = self._apply_noise(batch_x, noise_ctx)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Inference mode
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                
                preds.append(outputs)
                trues.append(batch_y)
                
                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'Test shape: {preds.shape}, {trues.shape}')
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'\n{"="*70}')
        print(f'Test Results:')
        print(f'   MSE:  {mse:.7f}')
        print(f'   MAE:  {mae:.7f}')
        print(f'   RMSE: {rmse:.7f}')
        print(f'   MAPE: {mape:.7f}')
        print(f'   MSPE: {mspe:.7f}')
        print(f'{"="*70}\n')
        
        f = open("result_jepa_vts.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}\n\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        print(f'Results saved to {folder_path}')
        
        return
