import json
import numpy as np
# Import Module
import os
import copy
import numpy as np
# from scipy.sparse.linalg import eigs
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA

from numpy.random import rand
import random
import csv
import pickle


import numpy as np

def hadamard_matrix(n):
    """ Generate an n x n Hadamard matrix using Sylvester's construction. """
    if n == 1:
        return np.array([[1]])
    else:
        H_n_minus_1 = hadamard_matrix(n // 2)
        return np.block([[H_n_minus_1, H_n_minus_1],
                         [H_n_minus_1, -H_n_minus_1]])

def binary_hadamard(n):
    """ Generate a binary Hadamard matrix (0-1) from the Hadamard matrix. """
    H = hadamard_matrix(n)
    return (H + 1) // 2  # Convert to binary (0-1)

def closest_power_of_two(n):
    """ Find the closest power of two greater than or equal to n. """
    return 2 ** np.ceil(np.log2(n)).astype(int)

def top_m_rows_hadamard(n, m):
    """ Return the first m rows of the binary Hadamard matrix with the most weight. """
    # Find the closest power of two greater than or equal to n
    closest_power = closest_power_of_two(n)
    
    # Generate the Hadamard matrix for the closest power of two
    H_full = binary_hadamard(closest_power)
    
    # Count the weight (number of ones) in each row
    weights = H_full.sum(axis=1)
    
    # Get the indices of the top m rows based on weight
    top_m_indices = np.argsort(weights)[-m:][::-1]  # Get indices of the top m weights
    
    # Select the top m rows
    top_m_rows = H_full[top_m_indices, :]
    
    # Randomly pick n columns from the top m rows
    num_cols = top_m_rows.shape[1]
    random_columns = np.random.choice(num_cols, n, replace=False)
    H_random = top_m_rows[:, random_columns]
    
    return H_random

# # Example usage
# n = 100  # Size of the Hadamard matrix (not necessarily a power of 2)
# m = 8  # Number of rows to select
# result = top_m_rows_hadamard(n, m)

# print("Top", m, "rows of the binary Hadamard matrix with the most weight after random column selection:")
# print(result)



def generate(n,m,k,measurment_density, mode):
    if mode =="random":
        flag=True
        while flag:
            I=np.random.binomial(1, measurment_density, size=(n,m))
            all_rows_have_one = all(any(cell == 1 for cell in row) for row in I)
            if all_rows_have_one:
                flag=False
            I=I
    elif mode=="Hadamard":
        I=top_m_rows_hadamard(n,m)
        I=I.T
    else:
        print("mode is not defined")
        
    
                
    incident_vector=np.zeros([n,1])
    incident_vector[np.random.permutation(n)[:k]]=1
    query_results=np.matmul(I.T,incident_vector)

     
        
        



    kk = np.sum(I, axis=0)
    I = np.copy(I[:,kk>1]);

    kk = np.sum(I, axis=1)
    I = np.copy(I[kk>0,:]);


    nn,mm=I.shape


    constraints=[]

    for i in range(mm):
        nodes = list(np.argwhere(I[:,i] > 0).T[0])
        constraints.append(nodes)

    header=[n,m]
    pth='./data/QGT/'+'n_'+str(n)+'m_'+str(m)+'k_'+str(k)+'p_'+str(measurment_density)+'.txt'

    f=open(pth, 'w')
    f.write(str(header[0]))
    f.write(' ')
    f.write(str(header[1]))
    f.write('\n')

    i=0
    query_num=0
    for cons in constraints:
        if len(cons)>1:
            for ns in cons:
                i+=1
                f.write(str(ns+1))
                if i< len(cons):
                    f.write(' ')
            f.write(' ')
            f.write(str(int(query_results[query_num][0])))
            query_num+=1
            
            f.write('\n')
        i=0
    

    #encoding location of ones

    locs = list(np.argwhere(incident_vector > 0))
    locations=[]
    for loc in locs:
        locations.append(loc[0])
    
    pth = './data/QGT_Truth/'+'n_'+str(n)+'m_'+str(m)+'k_'+str(k)+'p_'+str(measurment_density)+'.txt'
    f=open(pth, 'w')
    for loc in locs:
        f.write(str(loc[0]))
        f.write(' ')
    f.write('\n')
    cs_bound=k*np.log2(n/k)
    qgt_bound=2*k*np.log2(n/k)/np.log2(k)
    f.write('CS Bound: '+ str(cs_bound))
    f.write('\n')
    f.write('QGT Bound: '+str(qgt_bound))
    f.write('\n')

#     #Calculating the klog(n/k) and 2k log(n/k)/log(k) bounds

# cs_bound={}
# qgt_bound={}

#     for nn in n:
#         for kk in k:
#             cs_bound[(kk,nn)]=kk*np.log2(nn/kk)
#             qgt_bound[(kk,nn)]=2*kk*np.log2(nn/kk)/np.log2(k)




# #n=[100]
# k=[20]
# n=[500, 1000, 2000, 3000, 5000, 7000, 10000]
# #m=[2*n_i for n_i in n]
# m=[45]
# measurment_density=0.5

# for nn in n:
#     for kk in k:
#         for mm in m:
#             generate(nn,mm,kk)



    
