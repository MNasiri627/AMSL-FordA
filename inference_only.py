import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from train import conlstm_auto  


input_dim_x = 100
input_dim_y = 5
batch = 256
model_path = "/content/AMSL/models/"
data_path = "/content/AMSL/data/transform_dataset/train_dataset/"


def load_stack(path_prefix):
    raw = np.load(os.path.join(path_prefix, "data_raw_train.npy"))
    no = np.load(os.path.join(path_prefix, "data_no_train.npy"))
    ne = np.load(os.path.join(path_prefix, "data_ne_train.npy"))
    op = np.load(os.path.join(path_prefix, "data_op_train.npy"))
    pe = np.load(os.path.join(path_prefix, "data_pe_train.npy"))
    sc = np.load(os.path.join(path_prefix, "data_sc_train.npy"))
    ti = np.load(os.path.join(path_prefix, "data_ti_train.npy"))
    X = np.stack([raw, no, ne, op, pe, sc, ti], axis=1)  # (N, 7, 100, 5)
    return X.reshape((-1, 100, 5, 1))

X_train = load_stack(data_path)
initial_c = np.zeros((X_train.shape[0], 1))


print("Rebuilding model...")
model = conlstm_auto()
model.load_weights(os.path.join(model_path, 'model_best_weight.weights.h5'))
print("Model loaded.")


[predict_label8, predict_label9, predict_label1, predict_label2, predict_label3,
 predict_label4, predict_label5, predict_label6, predict_label7] = model.predict([X_train, initial_c], batch_size=batch, verbose=1)


class_loss = (predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] +
              predict_label4[:, 3] + predict_label5[:, 4] + predict_label6[:, 5] + predict_label7[:, 6])


np.savetxt(os.path.join(model_path, 'train_normal_loss_sparse.csv'), np.squeeze(predict_label9), delimiter=',')
np.savetxt(os.path.join(model_path, 'train_normal_loss_class.csv'), np.squeeze(class_loss), delimiter=',')
np.savetxt(os.path.join(model_path, 'train_normal_loss_sum_mse.csv'), np.squeeze(predict_label8), delimiter=',')


X_test_raw = np.load(os.path.join(data_path, "data_raw_test.npy"))
X_test_no = np.load(os.path.join(data_path, "data_no_test.npy"))
X_test_ne = np.load(os.path.join(data_path, "data_ne_test.npy"))
X_test_op = np.load(os.path.join(data_path, "data_op_test.npy"))
X_test_pe = np.load(os.path.join(data_path, "data_pe_test.npy"))
X_test_sc = np.load(os.path.join(data_path, "data_sc_test.npy"))
X_test_ti = np.load(os.path.join(data_path, "data_ti_test.npy"))

X_test = np.stack([X_test_raw, X_test_no, X_test_ne, X_test_op,
                   X_test_pe, X_test_sc, X_test_ti], axis=1)
X_test = X_test.reshape((-1, 100, 5, 1))
initial_c_test = np.zeros((X_test.shape[0], 1))

[predict_label8, predict_label9, predict_label1, predict_label2, predict_label3,
 predict_label4, predict_label5, predict_label6, predict_label7] = model.predict([X_test, initial_c_test], batch_size=batch, verbose=1)

class_loss = (
    predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] +
    predict_label4[:, 3] + predict_label5[:, 4] + predict_label6[:, 5] + predict_label7[:, 6]
)

np.savetxt(os.path.join(model_path, 'normal_loss_sparse.csv'), np.squeeze(predict_label9), delimiter=',')
np.savetxt(os.path.join(model_path, 'normal_loss_class.csv'), np.squeeze(class_loss), delimiter=',')
np.savetxt(os.path.join(model_path, 'normal_loss_sum_mse.csv'), np.squeeze(predict_label8), delimiter=',')


abnormal_path = "/content/AMSL/data/transform_dataset/test_dataset/"
ab_raw = np.load(os.path.join(abnormal_path, "data_raw_abnormal.npy"))
ab_no = np.load(os.path.join(abnormal_path, "data_no_abnormal.npy"))
ab_ne = np.load(os.path.join(abnormal_path, "data_ne_abnormal.npy"))
ab_op = np.load(os.path.join(abnormal_path, "data_op_abnormal.npy"))
ab_pe = np.load(os.path.join(abnormal_path, "data_pe_abnormal.npy"))
ab_sc = np.load(os.path.join(abnormal_path, "data_sc_abnormal.npy"))
ab_ti = np.load(os.path.join(abnormal_path, "data_ti_abnormal.npy"))

ab_data = np.stack([ab_raw, ab_no, ab_ne, ab_op, ab_pe, ab_sc, ab_ti], axis=1)
ab_data = ab_data.reshape((-1, 100, 5, 1))
initial_c_ab = np.zeros((ab_data.shape[0], 1))

[predict_label8, predict_label9, predict_label1, predict_label2, predict_label3,
 predict_label4, predict_label5, predict_label6, predict_label7] = model.predict([ab_data, initial_c_ab], batch_size=batch, verbose=1)

class_loss = (
    predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] +
    predict_label4[:, 3] + predict_label5[:, 4] + predict_label6[:, 5] + predict_label7[:, 6]
)

np.savetxt(os.path.join(model_path, 'abnormal_loss_sparse.csv'), np.squeeze(predict_label9), delimiter=',')
np.savetxt(os.path.join(model_path, 'abnormal_loss_class.csv'), np.squeeze(class_loss), delimiter=',')
np.savetxt(os.path.join(model_path, 'abnormal_loss_sum_mse.csv'), np.squeeze(predict_label8), delimiter=',')

print("All CSV files saved to:", model_path)
