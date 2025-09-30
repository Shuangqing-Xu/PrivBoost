from ast import Global
import pandas as pd
import numpy as np
import sys
import warnings
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from math import ceil
from math import log2
import multiprocessing
import os
warnings.filterwarnings('ignore')

T_f,M = np.uint32(10000),np.uint32(2**32)

def double_uint32(db_val): # mult by T_f
    unsigned_long_x = np.dtype(np.uint32) 
    long_long_x = np.int32(db_val*T_f)
    unsigned_long_x = np.uint32(long_long_x)
    return unsigned_long_x

def double_uint32_array(db_array):
    uint32_data = np.zeros_like(db_array, dtype = np.uint32)
    for i in range(0,db_array.shape[0]):
            uint32_data[i] = double_uint32(db_array[i])
    return uint32_data

def uint32_double(uint32_val): # div by T_f
    float_res = np.float64(np.int32(uint32_val))
    restore = float_res / T_f
    return restore

def time_uint32_double(uint32_val): # div by T_f
    float_res = np.float64(np.int32(uint32_val))
    restore = float_res / np.uint32(10000000)
    return restore

def uint32_double_array(uint32_array):
    restore = np.zeros_like(uint32_array, dtype = np.float64)
    for i in range(0,uint32_array.shape[0]):
            restore[i] = uint32_double(uint32_array[i])
    return restore


def non_sec_agg(n, client_gradient_histogram):
    agg_grad = 0
    for cur_client in range(n):
        quantization = double_uint32_array(client_gradient_histogram[cur_client])
        agg_grad += quantization
    return uint32_double_array(agg_grad)

def sec_agg(n, client_gradient_histogram):
    # prepare input vectors
    for cur_client in range(n):
        client_gradient_histogram[cur_client] = double_uint32_array(client_gradient_histogram[cur_client])
    np_client_gradient_histogram = np.array(client_gradient_histogram, dtype=np.uint32)
    np_client_gradient_histogram = np_client_gradient_histogram.flatten()
    np_client_gradient_histogram.tofile("./data_interaction/input_vectors.bin")

    # secure aggregation interacting with cpp function
    clientNum = n
    dim = client_gradient_histogram[0].shape[0]
    survRate = 1.0
    cmd = "./secagg/app "+str(clientNum)+" "+str(dim)+" "+str(survRate)
    os.system(cmd)
    read_pipe = np.fromfile("./data_interaction/pipe.bin", dtype=np.uint32)
    uint32_agg_grad = read_pipe[0:client_gradient_histogram[0].shape[0]]
    client_time = read_pipe[client_gradient_histogram[0].shape[0]]
    server_time = read_pipe[client_gradient_histogram[0].shape[0]+1]
    agg_grad = uint32_double_array(uint32_agg_grad)
    client_time = uint32_double(client_time)/1000 # /1000 is for turing 'ms' to 'second'
    server_time = uint32_double(server_time)/1000

    return agg_grad, client_time, server_time