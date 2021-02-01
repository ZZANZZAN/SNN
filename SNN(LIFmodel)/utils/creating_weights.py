import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def cr_W(kernel_size, num_feature_maps, num_full_con_lay):
	conv_kernel_layer2 = (0.5 - (-0.5))*np.random.random((kernel_size, kernel_size, num_feature_maps)) - 0.5
	conv_kernel_layer2.round(1)
	full_con_lay_W = (0.5 - (-0.5))*np.random.random((num_full_con_lay)) - 0.5
	full_con_lay_W.round(1)
	full_out_lay_W = (0.5 - (-0.5))*np.random.random((num_full_con_lay, num_feature_maps)) - 0.5
	full_out_lay_W.round(1)
	pool_kernel_l3 = np.array([[1,1],[1,1]])
	pool_kernel_l3.round(1)

	np.save('data_weight/conv_kernel_layer2',conv_kernel_layer2)
	np.save('data_weight/full_con_lay_W',full_con_lay_W)
	np.save('data_weight/full_out_lay_W',full_out_lay_W)
	np.save('data_weight/pool_kernel_l3',pool_kernel_l3)

def save_W(conv_kernel_layer2, full_con_lay_W, full_out_lay_W, pool_kernel_l3):
	np.save('data_weight/conv_kernel_layer2',conv_kernel_layer2)
	np.save('data_weight/full_con_lay_W',full_con_lay_W)
	np.save('data_weight/full_out_lay_W',full_out_lay_W)
	np.save('data_weight/pool_kernel_l3',pool_kernel_l3)
