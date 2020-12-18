
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_data(names=['madam', 'adam', 'sgd'], lr='large'):
    folder_path = './'
    paths = [os.path.join(folder_path, name) for name in names]
    paths = []
    for name in names:
        path = None       
        name = name + '_gan_g_' + lr + '_lr_test.pt'
        
        path = os.path.join(folder_path, name)
        paths.append(path)

    return [torch.load(path) for path in paths]

def plot_curves(names=['madam', 'adam', 'sgd'], 
         lr='large', labels=None, ylim=(0, 20), loc='upper right'):

    plt.figure()
    plt.ylim(ylim)
    curve_data = get_data(names, lr)
    for i in range(len(curve_data)):
        label = labels[i]
        loss = np.array(curve_data[i])
        if label == 'MAdam':
            plt.plot(loss, '-', label=label)
        else:
            plt.plot(loss, '--',label = label)
    
    plt.grid()
    plt.legend(fontsize=14, loc=loc)
    plt.title('Generator Loss ~ Epochs, {} learning rate'.format(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Generator Loss')
    plt.show()


if __name__ == '__main__':
    labels = ['MAdam', ' Adam', 'SGD']
    plot_curves(labels=labels, lr='xlarge')