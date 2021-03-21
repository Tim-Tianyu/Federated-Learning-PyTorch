from os import walk
import pickle
import matplotlib.pyplot as plt
import numpy as np

file_list = []
folder = '../save/objects/'
file_name = '../save/objects/'
OptCompare_MNIST = [
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[None]_Decay[None].pkl",
    "mnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[None].pkl",
    "mnist_cnn_30_iid[0]_Opt[adadelta]_Un[0]_Lr[None]_Decay[None].pkl",
    "mnist_cnn_30_iid[0]_Opt[adagrad]_Un[0]_Lr[None]_Decay[None].pkl"
    ]

OptCompare_FMNIST = [
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[None]_Decay[None].pkl",
    "fmnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[None].pkl",
    "fmnist_cnn_30_iid[0]_Opt[adadelta]_Un[0]_Lr[None]_Decay[None].pkl",
    "fmnist_cnn_30_iid[0]_Opt[adagrad]_Un[0]_Lr[None]_Decay[None].pkl"
    ]
OptCompare = ["sgd", "adam", "adadelta", "adagrad"]

SchedulerCompare_MNIST = [
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[step]_Decay[None].pkl",
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[None].pkl",
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[cosineAnnealing]_Decay[None].pkl"
]


SchedulerCompare_FMNIST = [
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[step]_Decay[None].pkl",
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[None].pkl",
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[cosineAnnealing]_Decay[None].pkl"
]

SchedulerCompare = ["Step", "Plateau", "Consine"]

for (dirpath, dirnames, filenames) in walk(folder):
    file_list.extend(filenames)
    break

for file in file_list:
    if not file.endswith('.pkl'):
        continue

    file_name = folder + file
    fig_name = '../save/' + file[0:-4]  + '.png'
    data = []
    print(file)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        train_loss = data[0]
        train_accuracy = data[1]
    f.close()

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig(fig_name)
    plt.close()

fig_name = '../save/OptCompare_MNIST.png'
plt.figure()
for i in range(4):
    file = OptCompare_MNIST[i]
    label = OptCompare[i]
    if not file.endswith('.pkl'):
        continue

    file_name = folder + file
    data = []
    print(file)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        train_loss = data[0]
        train_accuracy = data[1]
    f.close()

    plt.plot(range(len(train_loss)), train_loss, label=label)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.legend()
plt.savefig(fig_name)
plt.close()

fig_name = '../save/OptCompare_FMNIST.png'
plt.figure()
for i in range(4):
    file = OptCompare_FMNIST[i]
    label = OptCompare[i]
    if not file.endswith('.pkl'):
        continue

    file_name = folder + file
    data = []
    print(file)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        train_loss = data[0]
        train_accuracy = data[1]
    f.close()

    plt.plot(range(len(train_loss)), train_loss, label=label)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.legend()
plt.savefig(fig_name)
plt.close()

fig_name = '../save/SchedulerCompare_MNIST.png'
plt.figure()
for i in range(3):
    file = SchedulerCompare_MNIST[i]
    label = SchedulerCompare[i]
    if not file.endswith('.pkl'):
        continue

    file_name = folder + file
    data = []
    print(file)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        train_loss = data[0]
        train_accuracy = data[1]
    f.close()

    plt.plot(range(len(train_loss)), train_loss, label=label)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.legend()
plt.savefig(fig_name)
plt.close()

fig_name = '../save/SchedulerCompare_FMNIST.png'
plt.figure()
for i in range(3):
    file = SchedulerCompare_FMNIST[i]
    label = SchedulerCompare[i]
    if not file.endswith('.pkl'):
        continue

    file_name = folder + file
    data = []
    print(file)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        train_loss = data[0]
        train_accuracy = data[1]
    f.close()

    plt.plot(range(len(train_loss)), train_loss, label=label)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.legend()
plt.savefig(fig_name)
plt.close()