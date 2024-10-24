import networkx as nx
import numpy as np
import os














d=5

#path= "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/random_regular/reg_graph_100"
#path="/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxcut_data/stanford_data_p1/"
#os.chdir(path)
for n1 in range(1,10):
    n=500*n1
    a = nx.random_regular_graph(d, n, seed=None)
    #adj_a = {x: list(dict(a.adj)[x]) for x in dict(a.adj).keys()}

    #keys_s = sorted(adj_a.keys())

    #adj_a_s = {x: adj_a[x] for x in keys_s}

    L = int(d*n/2)

    G = np.zeros([L+1,2]).astype(np.int64)

    G [0,:] = [int(n), L]

    i=0

    G[1:,:]= a.edges

    G[1:,:] = G[1:,:] + np.ones([L,2])
    # for x , nodes in adj_a_s.items():
    #     for j in nodes:
    #         if j > x:
    #             i += 1
    #             G[i,:] = [int(x)+1,int(j)+1]

    name = 'G_reg_'+str(n)+'_'+str(d)+'.txt'
    path2 = "/Users/nasimeh/Documents/distributed_GCN-main-6/data/maxind_data/random_regular/d5/"

    np.savetxt(path2+name, G, fmt='%d')


