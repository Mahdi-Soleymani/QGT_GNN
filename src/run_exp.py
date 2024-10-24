import numpy

from src.data_reading import read_uf, read_stanford, read_hypergraph, read_hypergraph_task, read_NDC
from src.solver import  centralized_solver, centralized_solver_for
import logging
import os
import h5py
import numpy as np
import json
import timeit

#### loss BER for QGT
def BER(true,predict):

    false_negative=0
    for x in true:
        if x not in predict:
            false_negative+=1
        
    false_positive=0
    for y in predict:
        if y not in true:
            false_positive+=1

    error=false_negative+false_positive

    return (error, false_negative, false_positive)




def exp_centralized(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    # with h5py.File(params['res_path'], 'w') as f:
    all_outputs={}
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" or  params['data'] == "cliquegraph":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph" or params['data']=="QGT":############################################ MAHDI 
                constraints, header = read_hypergraph(path)
            elif params['data'] == "task":
                constraints, header = read_hypergraph_task(path)
            elif params['data'] == "NDC":
                constraints, header = read_NDC(path)
            else:
                log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

            res, res_th, outs, outs_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)
            
            #print(probs)
            #name= 'probs'+'_'+file_name+'.txt'
            #np.savetxt(name, probs)
            time = timeit.default_timer() - temp_time
            log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, training_time: {train_time}, mapping_time: {map_time}')
            print('\n \n')
            # print("Final loss:")
            # print(res)
            # print("Final loss with threshold:")
            # print(res_th)

            #truth=np.loadtxt(params['truth_path']+file_name)
            with open(params['truth_path']+file_name, 'r') as file:
                first_line = file.readline()
            truth = np.array([int(num) for num in first_line.split()])


            length=[]
            length_th=[]
            E=[]
            E_th=[]
            FP=[]
            FP_th=[]
            FN=[]
            FN_th=[]
    
            for out in outs:
                #outs_with_value_1=outs
                outs_with_value_1 = np.array([key for key, value in out.items() if value == 1])-1
                length.append(len(outs_with_value_1))
                e, fn, fp=BER(truth,outs_with_value_1)
                E.append(e)
                FN.append(fn)
                FP.append(fp)
            



            for outt in outs_th:
                outsth_with_value_1 = np.array([key for key, value in outt.items() if value == 1])-1
                # minus one is due to difference in indexing
                length_th.append(len(outsth_with_value_1))
                e,fn,fp=BER(truth,outsth_with_value_1)
                E_th.append(e)
                FN_th.append(fn)
                FP_th.append(fp)
            

            # Organize the lists into a dictionary
            data = {
                "length": length,
                "length_th": length_th,
                "E": E,
                "E_th": E_th,
                "FP": FP,
                "FP_th": FP_th,
                "FN": FN,
                "FN_th": FN_th,
                "Final_loss":res,
                "Final_loss_th":res_th
            }

            # # Save the dictionary to a JSON file
            # with open(f'{params["res_path"]}QGT_{os.path.splitext(file_name)[0]}.json', 'w') as json_file:
            #     json_file.write('{\n')  # Start of JSON object
            #     for key, value in data.items():
            #         json_file.write(f'    "{key}": {value},\n')  # Write each key-value pair
            #     json_file.seek(json_file.tell() - 2, 0)  # Remove the last comma
            #     json_file.write('\n}')  # End of JSON object
            #     #json.dump(data, json_file,  indent=None, separators=(',', ': '))

            # print(f'Data saved to {params["res_path"]}')

            all_outputs[f'{file_name}']=data
            

            if params['mode']=="partition":
                print(sum(outs))
                print(sum(outs_th))
                group={l: [] for l in range(params['n_partitions'])}
                for i in range(header['num_nodes']):
                    for l in range(params['n_partitions']):
                        if outs[i,l]==1:
                            group[l].append(i)
                with open(params['res_path'], 'w') as f:
                    json.dump(group, f)
    return all_outputs
            








def exp_centralized_for(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg":
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header, params, file_name)

                #name= 'probs'+'_'+file_name+'.txt'
                #np.savetxt(name, probs)
                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))
                if params['mode']=='maxind':
                    N = 200
                    print((np.average(res)) / (N*0.45537))
                f.create_dataset(f"{file_name}", data = res)




def exp_centralized_watermark(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" :
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                elif params['data'] == "task":
                    constraints, header = read_hypergraph_task(path)
                elif params['data'] == "NDC":
                    constraints, header = read_NDC(path)
                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                wat_len=16
                wat_type='MIS'
                wat_seed_value=100
                watermark_cons, watermark_nodes =generate_watermark(header['num_nodes'], wat_len, wat_type, wat_seed_value)
                res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_watermark(constraints, watermark_cons, watermark_nodes, header, params, file_name)

                #name= 'probs'+'_'+file_name+'.txt'
                #np.savetxt(name, probs)
                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(res)
                print(res_th)
                f.create_dataset(f"{file_name}", data = res)


import torch


def exp_centralized_for_multi(proc_id, devices, params):
    print("start to prepare for device")
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(master_ip="127.0.0.1", master_port="12345")
    torch.cuda.set_device(dev_id)
    TORCH_DEVICE = torch.device("cuda:" + str(dev_id))
    print("start to initialize process")
    torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=len(devices), rank=proc_id)
    print("start to train")

    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')

    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "arxiv":
                constraints, header = read_arxiv()
            else:
                log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, arxiv, and NDC.')

            # split the nodes into different devices
            total_nodes = header['num_nodes']
            cur_nodes = list(range(total_nodes * proc_id // len(devices), total_nodes * (proc_id + 1) // len(devices)))
            cur_nodes = [c + 1 for c in cur_nodes]
            inner_constraint = []
            outer_constraint = []
            for c in constraints:
                if c[0] in cur_nodes and c[1] in cur_nodes:
                    inner_constraint.append(c)
                elif (c[0] in cur_nodes and c[1] not in cur_nodes) or (c[0] not in cur_nodes and c[1] in cur_nodes):
                    outer_constraint.append(c)

            print("device", dev_id, "start to train")
            res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header,
                                                                                                params, file_name,
                                                                                             outer_constraint=outer_constraint)

            if res is not None:
                time = timeit.default_timer() - temp_time
                log.info(
                    f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))


                with h5py.File(params['res_path'], 'w') as f:
                    f.create_dataset(f"{file_name}", data=res)
