
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_data(names=['madam', 'adam', 'adabelief', 'adabound', 'sgd'], train=True):
    folder_path = './'
    paths = [os.path.join(folder_path, name) for name in names]
    paths = []
    for name in names:
        path = None
        if train:
            name = name + 'mobile_train.pt'
        else:
            name = name + 'mobile_test.pt'
        
        path = os.path.join(folder_path, name)
        paths.append(path)

    return [torch.load(path) for path in paths]

def plot_curves(names=['madam', 'adam', 'adabelief', 'adabound', 'sgd'], 
         train=True, labels=None, ylim=(55,100), loc='lower right'):

    plt.figure()
    plt.ylim(ylim)
    curve_data = get_data(names, train)
    for i in range(len(curve_data)):
        label = labels[i]
        acc = (1 - np.array(curve_data[i])) * 100
        print('%s acc: %2f%%' % (label, np.max(acc)))
        if label == 'MAdam':
            plt.plot(acc, '-', label=label)
        else:
            plt.plot(acc, '--',label = label)
    
    plt.grid()
    plt.legend(fontsize=14, loc=loc)
    curve = None
    if train:
        curve = 'Train'
    else:
        curve = 'Test'
    plt.title('Accuracy in {} set ~ Epochs'.format(curve))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Rate (%)')
    plt.show()


if __name__ == '__main__':
    labels = ['MAdam', ' Adam', 'AdaBelief', 'AdaBound', 'SGD']
    plot_curves(labels=labels, train=False)