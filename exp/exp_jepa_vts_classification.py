from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_JEPA_VTS_Classification(Exp_Basic):
    """
    JEPAVTS Classification Experiment
    Combines classification loss (CrossEntropy) with JEPA loss (MSE)
    """
    
    def __init__(self, args):
        super(Exp_JEPA_VTS_Classification, self).__init__(args)
        self.jepa_weight = getattr(args, 'jepa_weight', 1.0)
        print(f"✅ JEPA Weight: {self.jepa_weight}")

    def _build_model(self):
        # Model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        
        print(f"📊 Dataset info:")
        print(f"   seq_len: {self.args.seq_len}")
        print(f"   enc_in (features): {self.args.enc_in}")
        print(f"   num_class: {self.args.num_class}")
        
        # Model init - directly import JEPAVTS
        from models.JEPAVTS import Model as JEPAVTS
        model = JEPAVTS(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Forward pass (eval mode returns only class logits)
                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            class_losses = []
            jepa_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Forward pass (training mode returns class_logits, student_encoding, predicted_teacher_encoding)
                model_outputs = self.model(batch_x, padding_mask, None, None)
                
                if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                    # Single encoder: (class_logits, student_encoding, predicted_teacher_encoding)
                    class_logits, student_encoding, predicted_teacher_encoding = model_outputs
                    
                    # Get teacher encoding for JEPA loss
                    with torch.no_grad():
                        teacher_encoding = self.model.module.teacher(batch_x) if hasattr(self.model, 'module') else self.model.teacher(batch_x)
                    
                    # JEPA loss
                    jepa_loss = F.mse_loss(predicted_teacher_encoding, teacher_encoding)
                    
                elif isinstance(model_outputs, tuple) and len(model_outputs) == 2:
                    # Dual encoder: (class_logits, predicted_teacher_encoding)
                    class_logits, predicted_teacher_encoding = model_outputs
                    
                    # Get teacher encoding for JEPA loss
                    with torch.no_grad():
                        teacher_encoding = self.model.module.teacher(batch_x) if hasattr(self.model, 'module') else self.model.teacher(batch_x)
                    
                    # JEPA loss
                    jepa_loss = F.mse_loss(predicted_teacher_encoding, teacher_encoding)
                else:
                    # No JEPA output (shouldn't happen in training)
                    class_logits = model_outputs
                    jepa_loss = torch.tensor(0.0, device=self.device)

                # Classification loss
                class_loss = criterion(class_logits, label.long().squeeze(-1))
                
                # Combined loss
                total_loss = class_loss + self.jepa_weight * jepa_loss
                
                train_loss.append(total_loss.item())
                class_losses.append(class_loss.item())
                jepa_losses.append(jepa_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | "
                          f"total_loss: {total_loss.item():.5f} | "
                          f"class_loss: {class_loss.item():.5f} | "
                          f"jepa_loss: {jepa_loss.item():.5f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            class_loss_avg = np.average(class_losses)
            jepa_loss_avg = np.average(jepa_losses)
            
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.4f} (class: {class_loss_avg:.4f}, jepa: {jepa_loss_avg:.4f}) | "
                  f"Vali Loss: {vali_loss:.4f} Vali Acc: {val_accuracy:.4f} | "
                  f"Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}")
            
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f'accuracy: {accuracy}')
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write(f'accuracy: {accuracy}')
        f.write('\n')
        f.write('\n')
        f.close()
        return
