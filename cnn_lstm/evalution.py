import numpy as np
import matplotlib.pyplot as plt
import os
import cnn_lstm.utils as util
import re

# score initialization
valid_anomaly_score = np.zeros((util.valid_end_id - util.valid_start_id, 1))
test_anomaly_score = np.zeros((util.test_end_id - util.valid_end_id-util.step_max, 1))

# load the data from file
test_data_path = util.test_data_path
reconstructed_data_path = util.reconstructed_data_path
test_data_path = os.path.join(test_data_path, "test.npy")
reconstructed_data_path = os.path.join(reconstructed_data_path, "test_reconstructed.npy")
test_data = np.load(test_data_path)
test_data = test_data[:, -1, ...]  # only compare the last time step matrix with the reconstructed data
reconstructed_data = np.load(reconstructed_data_path)
print("The shape of test data is {}".format(test_data.shape))
print("The shape of reconstructed data is {}".format(reconstructed_data.shape))

valid_len = util.valid_end_id - util.valid_start_id

# compute the threshold, threshold = alpha * max{s(t)} , s(t) is the anomaly scores over validation period.
for i in range(valid_len):
	error = np.square(np.subtract(test_data[i, ..., 0], reconstructed_data[i, ..., 0]))
	num_anom = len(np.where(error > util.threhold)[0])
	valid_anomaly_score[i] = num_anom

max_valid_anom = np.max(valid_anomaly_score)
threshold = max_valid_anom * util.alpha
# threshold = 300

print("Max valid anom is %.2f" % max_valid_anom)
print("Threshold is %.2f" % threshold)

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# i = 376
# zz = test_data[i, ..., 0]
# test_matrix = test_data[i, ..., 0]
# sns.heatmap(test_matrix, cmap='rainbow',vmin=-0.75, vmax=1.25)
# plt.title("original matrix 376 ")
# plt.show()
#
# i = 376
# zz = test_data[i, ..., 0]
# reconstructed_matrix = reconstructed_data[i, ..., 0]
# sns.heatmap(test_matrix, cmap='rainbow',vmin=-0.75, vmax=1.25)
# plt.title("reconstructed matrix 376 ")
# plt.show()

# residual_matrix = np.square(np.subtract(test_data[i, ..., 0], reconstructed_data[i, ..., 0]))
# sns.heatmap(reconstructed_matrix, cmap='rainbow',vmin=0, vmax=1)
# plt.title(i)
# plt.show()

#
# compute the anomaly score in the test data.
for i in range(valid_len, len(test_data)):
	error = np.square(np.subtract(test_data[i, ..., 0], reconstructed_data[i, ..., 0]))
	num_anom = len(np.where(error > util.threhold)[0])
	test_anomaly_score[i - valid_len] = num_anom

# plot anomaly score curve and identification result
val_test_anomaly_score = np.concatenate((valid_anomaly_score, test_anomaly_score), axis=0)
anomaly_pos = np.zeros(5)
root_cause_gt = np.zeros((5, 3))
anomaly_span = [10, 30, 90]

# Read the test_anomaly.csv, each line behalf of an anomaly, the first is the position, the next three number is the
# root cause.
root_cause_f = open("../data/test_anomaly.csv", "r")

root_cause_gt = np.loadtxt(root_cause_f, delimiter=",", dtype=np.int32)
anomaly_pos = root_cause_gt[:, 0]
anomaly_pos = [(anomaly_pos[i]/util.gap_time-util.step_max-util.test_start_id) for i in range(5)]
print(anomaly_pos)
for i in range(5):
	root_cause_gt[i][0] = anomaly_pos[i]


fig, axes = plt.subplots()
test_num = util.test_end_id - util.test_start_id
plt.xticks(fontsize = 25)
plt.ylim((0, 120))
plt.yticks(np.arange(0, 121, 10), fontsize = 25)
plt.plot(val_test_anomaly_score, 'b', linewidth = 2)
threshold = np.full((test_num), max_valid_anom * util.alpha)
axes.plot(threshold, color = 'black', linestyle = '--',linewidth = 2)
for k in range(len(anomaly_pos)):
	# axes.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/util.gap_time, color='red', linewidth=2)
	axes.axvspan(anomaly_pos[k], anomaly_pos[k]+1, color='red', linewidth=2)

labels = [' ', '0', '200', '400', '600', '800', '1000']
axes.set_xticklabels(labels, rotation = 25, fontsize = 20)
plt.xlabel('Test Time', fontsize = 25)
plt.ylabel('Anomaly Score', fontsize = 25)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.25)
plt.title("MSCRED", size = 25)
plt.show()
