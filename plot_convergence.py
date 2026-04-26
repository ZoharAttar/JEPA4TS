"""
Plot training convergence curves from CSV logs.

Usage:
    # Compare two runs (apples-to-apples: pred loss, vali, test)
    python plot_convergence.py \
        --logs ./checkpoints/<timemixer_setting>/training_log.csv \
              ./checkpoints/<jepavts_setting>/training_log.csv \
        --labels TimeMixer JEPAVTS

    # Single run (shows all available columns)
    python plot_convergence.py \
        --logs ./checkpoints/<setting>/training_log.csv

    # Save to file instead of displaying
    python plot_convergence.py --logs ... --save convergence.pdf
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt


def read_log(path):
    """Read a training_log.csv and return {column_name: [values]}."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


def main():
    parser = argparse.ArgumentParser(description='Plot training convergence from CSV logs')
    parser.add_argument('--logs', nargs='+', required=True,
                        help='Paths to training_log.csv files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Display labels for each log (defaults to parent dir name)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to this path instead of displaying')
    args = parser.parse_args()

    logs = []
    for p in args.logs:
        logs.append(read_log(p))

    if args.labels:
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(p)) for p in args.logs]

    is_comparison = len(logs) > 1

    if is_comparison:
        # Apples-to-apples comparison: train pred loss, vali loss, test loss
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        for log, label in zip(logs, labels):
            epochs = log['epoch']
            # Use train_pred_loss if available (JEPAVTS), else train_loss (TimeMixer)
            pred_loss = log.get('train_pred_loss', log['train_loss'])
            axes[0].plot(epochs, pred_loss, '-o', markersize=3, label=label)
            axes[1].plot(epochs, log['vali_loss'], '-o', markersize=3, label=label)
            axes[2].plot(epochs, log['test_loss'], '-o', markersize=3, label=label)

        axes[0].set_title('Train Prediction Loss (MSE)')
        axes[1].set_title('Validation Loss (MSE)')
        axes[2].set_title('Test Loss (MSE)')

        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle('Convergence Comparison', fontsize=14, y=1.02)

    else:
        # Single run: show all columns
        log = logs[0]
        label = labels[0]
        epochs = log['epoch']

        has_jepa = 'train_jepa_loss' in log

        ncols = 4 if has_jepa else 3
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))

        pred_loss = log.get('train_pred_loss', log['train_loss'])
        axes[0].plot(epochs, pred_loss, '-o', markersize=3, color='tab:blue')
        axes[0].set_title('Train Prediction Loss')

        axes[1].plot(epochs, log['vali_loss'], '-o', markersize=3, color='tab:orange')
        axes[1].set_title('Validation Loss')

        axes[2].plot(epochs, log['test_loss'], '-o', markersize=3, color='tab:green')
        axes[2].set_title('Test Loss')

        if has_jepa:
            axes[3].plot(epochs, log['train_jepa_loss'], '-o', markersize=3, color='tab:red')
            axes[3].set_title('JEPA Alignment Loss')

        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Training Curves — {label}', fontsize=14, y=1.02)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
