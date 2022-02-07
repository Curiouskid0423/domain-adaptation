"""
A file for visualizing data (not part of training / inference)
"""
import matplotlib.pyplot as plt
import os

def no_axis_show(img, title=''):
    fig = plt.imshow(img, interpolation='nearest', cmap=None)
    # Do not show the axes in the images.
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)

titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 
    'television', 'dog', 'dolphin', 'spider']

if __name__ == '__main__':
    titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
    plt.figure(figsize=(12, 4))
    if not os.path.exists('./data/real_or_drawing/'): 
        print("Please download the dataset before visualizing.")
    # Visualize source domain
    for i in range(10):
        plt.subplot(1, 10, i+1)
        fig = no_axis_show(plt.imread(f'data/real_or_drawing/train_data/{i}/{500*i}.bmp'), title=titles[i])
    plt.show()
    # Visualize target domain
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        fig = no_axis_show(plt.imread(f'data/real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))
    plt.show()
    