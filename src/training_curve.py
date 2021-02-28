from os import walk
import pickle
import matplotlib.pyplot as plt
import numpy as np

file_list = []
folder = '../save/objects/'
file_name = '../save/objects/'
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
