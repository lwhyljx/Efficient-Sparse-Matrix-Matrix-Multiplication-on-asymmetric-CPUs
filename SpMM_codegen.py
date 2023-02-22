import copy
from dataclasses import dataclass
import os
from pathlib import Path
import random
import subprocess
from numpy import require


from code_template_short_gpu_new_fast import _CpuCodeTemplate, _CpuBodyCodeTemplate, _CpuBodyCodePruneWeightTemplate
# from code_template_short_gpu_new import _GpuCodeTemplate, _GpuBodyCodeTemplate, _GpuBodyCodePruneWeightTemplate
from generate_matrix import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity', required=True, type=int, default=50)
parser.add_argument('--M', required=True, type=int)
parser.add_argument('--K', required=True, type=int)
parser.add_argument('--N', required=True, type=int)
parser.add_argument('--numthread', required=True, type=int)


args = parser.parse_args()
M = 64      
K = 128
N = 256
TESAID = 0
sp_matrix = None
SPARSITY = args.sparsity
DATA_FOLDER = 'data'
CODE_FOLDER = 'codehub'
FILENAME = 'sparse_matrix.txt'
tiling_str = None
matrix_str = None
DEVICE_FUNC_PREFIX = None
device_funcs = []

BLOCK_SIZE_M = 0
BLOCK_SIZE_K = 0
BLOCK_SIZE_N = 0

def emit_block_device_func(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, blockIdx_x, DEVICE_FUNC_PREFIX):
    device_func = ""

    device_func_sig = f'void {DEVICE_FUNC_PREFIX}_device_func_blockIdx_x_{blockIdx_x}(float* input0, float *rC) {{\n'
    device_func += device_func_sig

    # device_func += "float* input0_tile = input0;\n"
    #num_K_tiles = int(K1 / K2)
    num_M_tiles= int(M1/M2)
    #for k_tile_idx in range(num_K_tiles):
        # device_func += f"input0_tile = input0 + blockIdx.x * {N2} + {k_tile_idx * K2 * N1};\n"
    device_func+=f" float local[{M2*N2}]={{0}};\n"
    for M_tile_idx in range(num_M_tiles):
        for m_step in range(M2):
            for n_step in range(N2):
                device_func+=f" local[{m_step*N2+n_step}]=0.000000e+00f;\n"
            for k_step in range(K1):
                tesa_value = tesa_matrix[(M_tile_idx * M2 + m_step) * K1 + k_step]
                sp_matrix_value = sp_matrix[(M_tile_idx * M2 + m_step) * K1 + k_step]
                if tesa_value != 0:
                    #for n_tile_idx in range(num_N_tiles):
                    for n_step in range(N2):
                        device_func+=f" local[{m_step*N2+n_step}]+={sp_matrix_value}f * input0[{(k_step) * N1+blockIdx_x*N2+n_step}];\n"
        for i in range(M2):
            for j in range(N2):
                device_func+=f" rC[{i*N1+blockIdx_x*N2+j}]+=local[{i*N2+j}];\n"
    device_func += f"}}\n"

    return device_func

def generate_dense_schema(M1, N1, K1, M2, N2, K2):
    global BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N
    BLOCK_SIZE_M = M2
    BLOCK_SIZE_K = K2
    BLOCK_SIZE_N = N2

def emit_finegrained_sparse_kernel_body(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, threadnum,DEVICE_FUNC_PREFIX, TRANSPOSE_OUTPUT=False, FUSE_ADD=False):
    device_funcs = []

    # emit device_funcs
    blockDim_x = int(N1 / N2)
    for blockIdx_x in range(blockDim_x):
        device_func = emit_block_device_func(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, blockIdx_x, DEVICE_FUNC_PREFIX)
        # print(device_func)
        device_funcs.append(device_func)
    if threadnum != 1:
        tile_thread=int(blockDim_x/threadnum)
        for i in range(threadnum):
            func_body = ""
            func_body+=f"void* threadfun_{i}(void* args) {{\n"
            func_body+=f" MY_ARGS* p=(MY_ARGS*)args;\n"
            #func_body+=f" set_cpu(p->tid);\n"
            func_body+=f" float* output0_tile=nullptr;\n"
            func_body+=f" float* input0=p->input0;\n"
            func_body+=f" float* output0=p->output0;\n"
            for blockIdx_x in range(tile_thread):
                func_body += f" {DEVICE_FUNC_PREFIX}_device_func_blockIdx_x_{i*tile_thread+blockIdx_x}(input0, output0);\n"
            func_body+=f" return nullptr;\n"
            func_body+=f"}}\n"
            device_funcs.append(func_body);  
    else:
        func_body = "";
        func_body +=f" void Matrixmatmul(float* input0, float* input1, float* output0){{\n"
        for blockIdx_x in range(blockDim_x):
                func_body += f" {DEVICE_FUNC_PREFIX}_device_func_blockIdx_x_{blockIdx_x}(input0, output0);\n"
        func_body+=f"}}\n"
        device_funcs.append(func_body);  
    # # write back output0_local
    # func_body += f"float *output0_tile = output0 + (blockIdx.x * {M2} + threadIdx.x) * {N1} + blockIdx.y * {N2};\n"
    # func_body += f"float4 *output0_tile_f4 = reinterpret_cast<float4*>(output0_tile);\n"
    # func_body += f"float4 *output0_local_f4 = reinterpret_cast<float4*>(output0_local);\n"
    # if FUSE_ADD:
    #     func_body += f"float4 *bias_f4 = reinterpret_cast<float4*>(input2+blockIdx.y*{N2});\n"
    #     func_body += f"float4 bias_f4_local;\n"
    #     for item in range(int(N2/4)):
    #         func_body += f"bias_f4_local = bias_f4[{item}];\n"
    #         func_body += f"output0_local_f4[{item}].x += bias_f4_local.x;\n"
    #         func_body += f"output0_local_f4[{item}].y += bias_f4_local.y;\n"
    #         func_body += f"output0_local_f4[{item}].z += bias_f4_local.z;\n"
    #         func_body += f"output0_local_f4[{item}].w += bias_f4_local.w;\n"
    
    # for item in range(int(N2/4)):
    #     func_body += f"output0_tile_f4[{item}] = output0_local_f4[{item}];\n"

    return device_funcs

def code_gen(M1, N1, K1, M2, N2, K2, threadnum,write_file=False):
    global sp_matrix, device_funcs, DEVICE_FUNC_PREFIX
    sp_matrix = load_tesa_matrix_float(FILENAME)
    generate_dense_schema(M1, N1, K1, M2, N2, K2)
    tesa_matrix = sp_matrix
    device_funcs = emit_finegrained_sparse_kernel_body(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, threadnum,DEVICE_FUNC_PREFIX, False, False)

    device_func_code = ""
    for device_func in device_funcs:
        device_func_code += device_func

    pthread_creat=""
    if threadnum!=1:
        pthread_creat+=f" pthread_t* thread_handles=new pthread_t[{threadnum}];\n"
        pthread_creat+=f" MY_ARGS* args=new MY_ARGS[{threadnum}];\n"
        for i in range(threadnum):
            pthread_creat+=f" args[{i}].tid={i*2};\n"
            pthread_creat+=f" args[{i}].input0=d_B.getData();\n"
            pthread_creat+=f" args[{i}].output0=d_C.getData();\n"

    pthread_run=""
    if threadnum!=1:
        for i in range(threadnum):
            pthread_run+=f" pthread_create(&thread_handles[{i}],NULL,threadfun_{i},(void *)(&args[{i}]));\n"
        pthread_run+=f"for(int k=0;k<{threadnum};k++){{\n"
        pthread_run+=f"pthread_join(thread_handles[k],NULL);\n"
        pthread_run+=f"}}\n"
    else:
        pthread_run+=f" Matrixmatmul(d_B.getData(), d_A.getData(), d_C.getData());\n"
        
    full_code = _CpuCodeTemplate.format(device_func_code,M, N, K, matrix_str, pthread_creat,pthread_run,pthread_run,threadnum,M2,K2,N2)

    code_file_name = None
    if write_file:
        code_file_name = f'{CODE_FOLDER}/_generated_cpu_embed_{DEVICE_FUNC_PREFIX}_{tiling_str}_{threadnum}.cpp'
        with open(code_file_name, 'w') as fp:
            fp.write(full_code)

    return code_file_name


if __name__ == '__main__':
    # ta = nni.get_next_parameter()
    M2=int(args.M)
    N2=int(args.N)
    K2=int(args.K)
    ta = {'M2': M2, 'N2': N2, 'K2': K2}
    tiling_str = f'{ta["M2"]}_{ta["N2"]}'
    matrix_str = f'{M}_{K}_{N}'
    os.system(f'mkdir {CODE_FOLDER}')
    os.system(f'cp dev_array.h {CODE_FOLDER}/dev_array.h')
    os.system(f'mkdir {DATA_FOLDER}')
    FILENAME = f'{DATA_FOLDER}/sparse_matrix_{matrix_str}.txt'
    # FILENAME = f'{DATA_FOLDER}/sparse_matrix_{tiling_str}.txt'
    # FILENAME = f'{DATA_FOLDER}/mlp_mnist/sparse_matrix_{matrix_str}_fc4.txt'
    # FILENAME = f'{DATA_FOLDER}/mlp_mnist/sparse_matrix_{matrix_str}_{SPARSITY}_mix.txt'
    # FILENAME = f'../models/bert_finegrained_onnx_with_tesa/Constants/Dot_4096_768_768_TESAID_10.csv'
    generate_sparse_matrix_float(M, K, SPARSITY, FILENAME)
    DEVICE_FUNC_PREFIX = f'sparse_matrix_{matrix_str}_{SPARSITY}'
    threadnum=int(args.numthread)
    code_file = code_gen(M, N, K, ta["M2"], ta["N2"], ta["K2"], threadnum,write_file=True)
    #avg_latency, latencys = profile_gpu_code(code_file, tiling_str)
    # nni.report_final_result({'default': avg_latency, 'all_latency': latencys})
    # nni.report_final_result(avg_latency)