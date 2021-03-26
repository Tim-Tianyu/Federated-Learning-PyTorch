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

def produce_compare_fig(fig_name, file_list, lable_list):
    fig_name = '../save/' + fig_name +'.png'
    plt.figure()
    for i in range(len(file_list)):
        file = file_list[i]
        label = lable_list[i]
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

Decay_compare = ["plateau", "step", "expontial"]

Decay_compare_adam_mnist = [
    "mnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[loss].pkl",
    "mnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[commu].pkl",
    "mnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[None]_exp_decay.pkl"
]

produce_compare_fig("DecayCompare_Adam_MNIST", Decay_compare_adam_mnist, Decay_compare)

Decay_compare_plateau_mnist = [
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[loss].pkl",
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[commu].pkl",
    "mnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[exponetial].pkl"
]

produce_compare_fig("DecayCompare_Plateau_MNIST", Decay_compare_plateau_mnist, Decay_compare)

Decay_compare_adam_fmnist = [
    "fmnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[loss].pkl",
    "fmnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[commu].pkl",
    "fmnist_cnn_30_iid[0]_Opt[adam]_Un[0]_Lr[None]_Decay[None]_exp_decay.pkl"
]

produce_compare_fig("DecayCompare_Adam_FMNIST", Decay_compare_adam_fmnist, Decay_compare)

Decay_compare_plateau_fmnist = [
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[loss].pkl",
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[commu].pkl",
    "fmnist_cnn_30_iid[0]_Opt[sgd]_Un[0]_Lr[reduceOnPlateau]_Decay[exponetial].pkl"
]

produce_compare_fig("DecayCompare_Plateau_FMNIST", Decay_compare_plateau_fmnist, Decay_compare)
