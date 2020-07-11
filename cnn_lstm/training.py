import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp
from cnn_lstm.MSCRED_torch import MSCRED
import timeit
import pandas as pd
import os, sys
from cnn_lstm import utils
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def train(model,**kwargs):
    # data input
    matrix_data_path = utils.train_data_path + "train.npy"
    matrix_gt = np.load(matrix_data_path)
    matrix_gt = matrix_gt.transpose((0,1,4,2,3))
    t_matrix_gt = t.from_numpy(matrix_gt)
    t_matrix_gt = t.tensor(t_matrix_gt, dtype=t.float32)
    # model definition
    model = model
    # loss function
    loss_func = t.nn.MSELoss()
    # optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=utils.learning_rate)

    # training process
    start = timeit.default_timer()
    loss_all = []
    for i in range(starting_iter, starting_iter + utils.training_iters):
        print("training iteration " + str(i) + "...")
        for idx in range(len(matrix_gt)):
            matrix_gt_per = t_matrix_gt[idx]
            reconstruct = model(matrix_gt_per).squeeze(0)
            loss = loss_func(reconstruct, matrix_gt_per[-1])
            loss_all.append(loss.item())
            print("mse of last train data: " + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    stop = timeit.default_timer()
    print(str(utils.training_iters) + " iterations training finish, use time: " + str(stop - start))
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i}
    t.save(state, utils.model_path+'mscred.pth')
    # plot the training mse loss curve
    plt_data = pd.DataFrame(np.asarray(loss_all))
    ax = sns.lineplot(data=plt_data)
    plt.savefig("torch_training_loss.jpg")
    plt.show()


@t.no_grad()
def test(dir):
    # data input
    # Read the data from test file.
    matrix_data_path = utils.test_data_path + "test.npy"
    matrix_gt = np.load(matrix_data_path)
    matrix_gt = matrix_gt.transpose((0,1,4,2,3))
    t_matrix_gt = t.from_numpy(matrix_gt)
    t_matrix_gt = t.tensor(t_matrix_gt, dtype=t.float32)
    # load model
    checkpoint = t.load(dir)
    model = MSCRED()
    model.load_state_dict(checkpoint['net'])
    optimizer = t.optim.Adam(model.parameters(), lr=utils.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # loss function
    loss_func = t.nn.MSELoss()

    result_all = []
    reconstructed_data_path = utils.reconstructed_data_path
    if not os.path.exists(reconstructed_data_path):
        os.makedirs(reconstructed_data_path)
    print("model test: generate recontrucuted matrices" + "...")
    loss_all = []
    for idx in range(len(matrix_gt)):
        matrix_gt_per = t_matrix_gt[idx]
        reconstructed_matrix = model(matrix_gt_per).squeeze(0)
        loss = loss_func(reconstructed_matrix, matrix_gt_per[-1])
        result_all.append(reconstructed_matrix.numpy())
        loss_all.append(loss.item())
        print("mse of last test data: " + str(loss.item()))

    result_all = np.asarray(result_all).reshape((-1, 3, 30, 30))
    result_all = result_all.transpose((0, 2, 3, 1))
    print(result_all.shape)
    np.save(reconstructed_data_path+"test_reconstructed.npy", result_all)

    # plot the testing mse loss curve
    print(np.where(np.asarray(loss_all) > 0.005))
    plt_data = pd.DataFrame(np.asarray(loss_all))
    ax = sns.lineplot(data=plt_data)
    plt.savefig("torch_testing_loss.jpg")
    plt.show()

starting_iter = 0
if utils.train_test_label == 1:
    model = MSCRED()
    train(model)

if utils.train_test_label == 0:
    test(utils.model_path+'mscred.pth')