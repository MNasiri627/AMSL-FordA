import csv
import numpy as np
from prettytable import PrettyTable

# === Load training normal loss (for threshold)
csv_file = csv.reader(open('/content/AMSL/models/train_normal_loss_sum_mse.csv', 'r'))
data1 = [float(row[0]) for row in csv_file]
bb_t = np.array(data1)
thre = np.percentile(bb_t, 98)

# === Load normal test loss
csv_file = csv.reader(open('/content/AMSL/models/normal_loss_sum_mse.csv', 'r'))
data = [float(row[0]) for row in csv_file]
bb = np.array(data)
print("Normal test shape:", bb.shape)

normal_true = bb[bb[:] < thre]
print('normal_sum :' ,bb.shape[0], 'normal_true:', normal_true.shape[0])

# === Load abnormal test loss
csv_file = csv.reader(open('/content/AMSL/models/abnormal_loss_sum_mse.csv', 'r'))
data = [float(row[0]) for row in csv_file]
bb1 = np.array(data)
print("Abnormal test shape:", bb1.shape)

abnormal_true = bb1[bb1[:] >= thre]
print('abnormal_sum :' ,bb1.shape[0], 'abnormal_true:', abnormal_true.shape[0])

# === Metrics
normal_true = normal_true.shape[0]
fatigue_true = abnormal_true.shape[0]
recerr2 = bb.shape[0]
recerr3 = bb1.shape[0]

acc = (normal_true + fatigue_true) / (recerr2 + recerr3)
precision_n = normal_true / (normal_true + recerr3 - fatigue_true)
precision_a = fatigue_true / (fatigue_true + recerr2 - normal_true)
pre_avg = (precision_a + precision_n) / 2
recall_n = normal_true / recerr2
recall_a = fatigue_true / recerr3
recall_avg = (recall_a + recall_n) / 2
F1 = 2 * (pre_avg * recall_avg) / (pre_avg + recall_avg)

# === Print table
x = PrettyTable(["acc", "pre_normal", "pre_abn", "pre_avg", "recall_normal", "recall_abn", "recall_avg", "F1"])
x.add_row([acc, precision_n, precision_a, pre_avg, recall_n, recall_a, recall_avg, F1])
print(x)