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
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        # Optional: VisionTS++ prediction (forecast) regularizer.
        #   L = MSE(student, y_true) + weight * MSE(student, visionts++_forecast)
        # The VisionTS++ forecast is frozen (no grad), computed on-the-fly from the
        # SAME look-back window the student sees, in colorized-multivariate (MV) mode.
        # Off by default -> plain training is unchanged unless --vts_pred_reg is set.
        self.vts_reg = getattr(args, 'vts_pred_reg', False)
        # Teacher forecast source: 'visionts' (VisionTS++, on-the-fly) or 'timesfm'
        # (precomputed TimesFM forecasts loaded by hash). TimesFM is the non-vision
        # ablation: same regularizer, a TS foundation model instead of a visual one.
        # 'timesfm' = univariate (CI) TimesFM forecasts; 'timesfm_mv' = joint
        # multivariate TimesFM forecasts (all channels in one context).
        # 'visionts_long' = precomputed VisionTS++ MV forecasts from a LONGER teacher
        # look-back (--vts_teacher_seq_len) ending at the same step as the student's
        # 96-step window; windows without enough history have no entry and are masked.
        self.vts_source = getattr(args, 'vts_pred_source', 'visionts')
        if self.vts_reg and (self.vts_source.startswith('timesfm') or self.vts_source == 'visionts_long'):
            import os as _os
            stem = _os.path.splitext(_os.path.basename(args.data_path))[0]
            if self.vts_source == 'visionts_long':
                T = getattr(args, 'vts_teacher_seq_len', 0)
                P = getattr(args, 'vts_pred_periodicity', 24)
                self.fc_cache = _os.path.join(
                    args.root_path, f"visionts_long_forecast_{stem}_T{T}_P{P}_{args.pred_len}")
                hint = (f"utils/precompute_visionts_long_forecast.py --dataset {stem} "
                        f"--pred_len {args.pred_len} --teacher_seq_len {T} --periodicity {P}")
            else:
                prefix = "timesfm_mv_forecast" if self.vts_source == 'timesfm_mv' else "timesfm_forecast"
                mode = 'mv' if self.vts_source == 'timesfm_mv' else 'ci'
                self.fc_cache = _os.path.join(args.root_path, f"{prefix}_{stem}_{args.pred_len}")
                hint = (f"utils/precompute_timesfm_forecast.py --dataset {stem} "
                        f"--pred_len {args.pred_len} --mode {mode}")
            if not _os.path.isdir(self.fc_cache):
                raise ValueError(f"Teacher forecast cache not found: {self.fc_cache}\n   Run {hint} first!")
            self.vts_reg_weight = getattr(args, 'vts_pred_reg_weight', 1.0)
            n = len([f for f in os.listdir(self.fc_cache) if f.endswith('.npy')])
            print(f"[pred-reg] source={self.vts_source}, weight={self.vts_reg_weight}, "
                  f"cache={self.fc_cache} ({n} forecasts)")
        elif self.vts_reg:
            from visionts import VisionTSpp
            arch = getattr(args, 'vts_pred_arch', 'mae_base')
            ckpt_dir = getattr(args, 'vts_pred_ckpt_dir', './hf_models/VisionTSpp')
            ckpt_path = os.path.join(ckpt_dir, f"visiontspp_{arch.split('_')[1]}.ckpt")
            # Rendering of the teacher forecast: 'mv' = colorized multivariate
            # (color=True, all vars in one image), 'ci' = per-variable grayscale
            # (color=False, each variable forecast on its own image).
            self.vts_render = getattr(args, 'vts_pred_render', 'mv')
            color = (self.vts_render == 'mv')
            self.vts_teacher = VisionTSpp(
                arch=arch, finetune_type='none', ckpt_path=ckpt_path, load_ckpt=True,
                quantile=True, clip_input=0, complete_no_clip=False, color=color,
            ).to(self.device).eval()
            for p in self.vts_teacher.parameters():
                p.requires_grad = False
            self.vts_P = getattr(args, 'vts_pred_periodicity', 24)
            self.vts_teacher.update_config(
                context_len=args.seq_len, pred_len=args.pred_len,
                periodicity=self.vts_P, norm_const=0.4, align_const=0.4)
            self.vts_reg_weight = getattr(args, 'vts_pred_reg_weight', 1.0)
            print(f"[VTS pred-reg] weight={self.vts_reg_weight}, periodicity={self.vts_P}, "
                  f"render={self.vts_render} (color={color}), context={args.seq_len}")

    def _vts_forecast(self, batch_x, f_dim):
        """Frozen VisionTS++ forecast target for the student's input window.

        batch_x: [B, seq_len, N] (scaled, on device) -> [B, pred_len, N'] detached.
        'mv': feed all vars at once (one colorized image).
        'ci': fold vars into the batch, forecast each variable on its own image.
        Forced to fp32 (autocast disabled) for a stable teacher target.
        """
        if self.vts_source.startswith('timesfm'):
            return self._cached_forecast(batch_x, f_dim)[0]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            x = batch_x.float()
            if self.vts_render == 'ci':
                B, L, N = x.shape
                xi = x.permute(0, 2, 1).reshape(B * N, L, 1)   # [B*N, L, 1]
                y = self.vts_teacher.forward(xi)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                y = y[..., 0].reshape(B, N, -1).permute(0, 2, 1)  # [B, pred_len, N]
            else:
                y = self._vts_mv_grouped(x)
            y = y[:, -self.args.pred_len:, f_dim:]
        return y.detach()

    def _vts_mv_grouped(self, x, max_images=256):
        """MV forecast, splitting wide datasets into channel groups.

        MV packs every variable into one 224-px image (224 // N rows each), which
        breaks once N is large (electricity 321, traffic 862 -> 0 rows). If N exceeds
        --vts_pred_mv_group (default 21), split the channels into groups of that size,
        render each group as its own MV image, and stitch the forecasts back. N <= group
        is a single image, so ETT (7) / exchange (8) / weather (21) are unchanged.
        """
        def fwd(inp):
            out = self.vts_teacher.forward(inp)
            return out[0] if isinstance(out, (list, tuple)) else out  # quantile median

        B, L, N = x.shape
        G = getattr(self.args, 'vts_pred_mv_group', 21)
        if N <= G:
            return fwd(x)
        n_chunks = -(-N // G)
        pad = n_chunks * G - N
        if pad:  # replicate the last channel (avoids zero-variance series), dropped later
            x = torch.cat([x, x[..., -1:].expand(B, L, pad)], dim=-1)
        xc = x.reshape(B, L, n_chunks, G).permute(0, 2, 1, 3).reshape(B * n_chunks, L, G)
        ys = [fwd(xc[s:s + max_images]) for s in range(0, xc.shape[0], max_images)]
        y = torch.cat(ys, dim=0)                                  # [B*n_chunks, P, G]
        P = y.shape[1]
        y = y.reshape(B, n_chunks, P, G).permute(0, 2, 1, 3).reshape(B, P, n_chunks * G)
        return y[..., :N]

    def _cached_forecast(self, batch_x, f_dim, allow_missing=False):
        """Load precomputed teacher forecasts (TimesFM / long-context VisionTS++) by
        the md5 hash of the student's input window.

        batch_x: [B, seq_len, N] -> (y [B, pred_len, N'] detached, mask [B] float).
        allow_missing=False raises on a missing entry; True returns zeros there with
        mask 0 (used by 'visionts_long', whose earliest windows have no teacher).
        """
        import hashlib
        x = batch_x.detach().cpu().numpy().astype(np.float32)   # [B, L, N]
        outs, mask = [], []
        shape = None
        for i in range(x.shape[0]):
            h = hashlib.md5(x[i].astype(np.float32).tobytes()).hexdigest()[:16]
            path = os.path.join(self.fc_cache, f"{h}.npy")
            if os.path.exists(path):
                f = torch.from_numpy(np.load(path))                # [pred_len, N]
                shape = f.shape
                outs.append(f); mask.append(1.0)
            elif allow_missing:
                outs.append(None); mask.append(0.0)
            else:
                raise FileNotFoundError(
                    f"Teacher forecast not found: {path}\n"
                    f"Run the matching precompute script for this dataset/pred_len.")
        if shape is None:                                          # whole batch masked
            shape = (self.args.pred_len, x.shape[2])
        outs = [o if o is not None else torch.zeros(shape) for o in outs]
        y = torch.stack(outs, dim=0).to(batch_x.device).float()   # [B, pred_len, N]
        m = torch.tensor(mask, device=batch_x.device)
        return y[:, -self.args.pred_len:, f_dim:].detach(), m

    def _reg_term(self, outputs, batch_x, f_dim):
        """Prediction-regularizer MSE between the student forecast and the teacher's.
        For 'visionts_long', samples without a teacher forecast are masked out."""
        if self.vts_source == 'visionts_long':
            y, m = self._cached_forecast(batch_x, f_dim, allow_missing=True)
            if m.sum() == 0:
                return outputs.sum() * 0.0
            m = m.view(-1, 1, 1).to(outputs.dtype)
            sq = (outputs - y.to(outputs.dtype)) ** 2 * m
            return sq.sum() / (m.sum() * outputs.shape[1] * outputs.shape[2])
        return nn.functional.mse_loss(outputs, self._vts_forecast(batch_x, f_dim).to(outputs.dtype))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

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

        csv_path = os.path.join(path, 'training_log.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'vali_loss', 'test_loss'])

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if self.vts_reg:
                            loss = loss + self.vts_reg_weight * self._reg_term(outputs, batch_x, f_dim)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if self.vts_reg:
                        loss = loss + self.vts_reg_weight * self._reg_term(outputs, batch_x, f_dim)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            csv_writer.writerow([epoch + 1, train_loss, vali_loss, test_loss])
            csv_file.flush()

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        csv_file.close()
        print(f"📄 Training log saved to {csv_path}")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _load_transfer_state_dict(self, state_dict):
        """Load a source checkpoint tolerating shape-mismatched tensors.

        Keeps only tensors whose shape matches the current (target) model; keys that
        differ (e.g. enc_in-sized RevIN affine when the target has a different number
        of variables) are skipped and keep their initialization. Uses strict=False so
        the skipped keys are simply left as 'missing'.
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
            print('loading model from {}'.format(ckpt_path))
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if transfer_ckpt:
                # Cross-dataset transfer: the target may have a different number of
                # variables than the source. Drop checkpoint tensors whose shape does
                # not match the (target-enc_in) model — e.g. TimeMixer's RevIN
                # normalize_layers.*.affine_{weight,bias} of shape [enc_in]. Those are
                # per-source-channel learned scale/shift that do not transfer across
                # datasets; leaving them at their identity init keeps RevIN's instance
                # (per-window) mean/std normalization intact.
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
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x = self._apply_noise(batch_x, noise_ctx)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
