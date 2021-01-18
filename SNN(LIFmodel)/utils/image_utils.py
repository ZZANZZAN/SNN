import random
import numpy as np
import matplotlib.pyplot as plt

import mnist 
mndata = MNIST('./dir_with_mnist_data_files')
images, labels = mndata.load_training()


def get_next_image(index=0, pick_random = False, display=True):
    if pick_random:
        index = random.randint(0, len(images)-1)
    image = images[index]
    label = labels[index]
    if display:
        print('Label: {}'.format(label))
        print(mndata.display(image))
    image = np.asarray(image).reshape((28,28))
    image_norm = (image * 255.0/image.max()) / 255.
    return image_norm, label


def graph_retinal_image(image, stride):
    fig = plt.figure()

    len_x, len_y = image.shape
    x_max = int(len_x/stride[0])
    y_max = int(len_y/stride[0])
    print('Convolution Dimensions: x={} / y={}'.format(x_max, y_max))
    x_count, y_count = 1, 1

    for y in range (0, len_y, stride[0]):
        x_count = 1
        for x in range(0, len_x, stride[0]):
            x_end = x + stride[0]
            y_end = y + stride[0]
            kernel = image[y:y_end, x:x_end]
            #orientation = s1(kernel)
            a = fig.add_subplot(y_max, x_max, (y_count-1)*x_max+x_count)
            a.axis('off')
            plt.imshow(kernel, cmap="gray")
            x_count += 1
        y_count += 1
    plt.show()


