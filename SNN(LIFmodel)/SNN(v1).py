import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import redis
from mnist import MNIST

from neurons import LIFNeuron as LIF
from utils import graph_results as graph, image_utils

T = 20   
dt = 0.01 
time = int(T / dt)
_T = 1
stride = (4,2)   
stride_size = stride[0] + stride[1]
kernel_size = 5
debug=False

#image, label = image_utils.get_next_image(pick_random = True)  

#image_utils.graph_retinal_image(image, stride)

len_x = 28
len_y = 28

# Инициализация первого (входного) слоя
neurons_l1 = []
print('Creating {} x {} neurons layer 1'.format(len_x, len_y))
for y in range (0, len_y, 1):
    neuron_row=[]
    for x in range(0, len_x, 1):
        neuron_row.append(LIF.LIFNeuron(neuron_label="L1:{}/{}".format(y,x), debug=debug))
    neurons_l1.append(neuron_row)

for y in range(0, len_y, 1):
    for x in range(0, len_x, 1):
        stimulus = np.full((time), image[y,x])
        neurons_l1[y][x].spike_generator(stimulus)

# Инициализация второго (сверточного) слоя
conv_kernel_layer2_1 = np.random.random((kernel_size,kernel_size))
conv_kernel_layer2_1.round(1)
for x in range(kernel_size):
	for y in range(kernel_size):
		if conv_kernel_layer2_1[x][y] > 0.5: conv_kernel_layer2_1[x][y] = 0.5
		if conv_kernel_layer2_1[x][y] < -0.5: conv_kernel_layer2_1[x][y] = -0.5

conv_kernel_layer2_2 = np.random.random((kernel_size,kernel_size))
conv_kernel_layer2_2.round(1)
for x in range(kernel_size):
	for y in range(kernel_size):
		if conv_kernel_layer2_2[x][y] > 0.5: conv_kernel_layer2_2[x][y] = 0.5
		if conv_kernel_layer2_2[x][y] < -0.5: conv_kernel_layer2_2[x][y] = -0.5

neurons_l2 = []
len_x_l2 = int(len_x - kernel_size + 1)
len_y_l2 = int(len_y - kernel_size + 1)

for y in range (0, len_y_l2, 1):
    neuron_row=[]
    for x in range(0, len_x_l2, 1):
        neuron_row.append(LIF.LIFNeuron(neuron_label="L2:{}/{}".format(y,x), debug=debug))
    neurons_l2.append(neuron_row)