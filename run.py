from src.run_exp import exp_centralized,  exp_centralized_for
from src.solver import QUBO_solver
import json
import numpy as np
import sys
from src.QGT_Gen import generate
import glob
import os
from datetime import datetime

# Function to convert numpy types to native types
def convert_numpy_types(obj):
    
    if isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.generic):  # This covers all numpy scalar types
        return obj.item()  # Convert numpy scalar to its native type
    elif isinstance(obj, np.ndarray):  # Check for numpy arrays
        return obj.tolist()  # Convert numpy array to list
    else:
        return obj




#####generating samples

k=[10]
n=[100]
m=np.arange(20,80,2)
measurment_density=0.2
#### mode how measurement matrix is generated could be random or Hadamard
mode="random"
#mode="Hadamard"
for nn in n:
    for kk in k:
        for mm in m:
            generate(nn,mm,kk, measurment_density,mode)

#####################################################
attribute_name="f"
attribute_values=np.arange(5,55,5,dtype=int)
#####################################################




with open('configs/QGT.json') as f:
   params = json.load(f)
all_results={}
### The first entry is the parameter we sweep
all_results["swept_value"]="attribute_name"
### The second entry in all_result saves the config file that is the same for all
all_results["parameters"]=params

for val in attribute_values:
   params[f'{attribute_name}']=val
   print(f'working on {attribute_name}={val}')
   result_dict=exp_centralized(params)
   all_results[f'{val}']=result_dict

all_results_sorted_by_file_name={}
### The first entry is the parameter we sweep
all_results_sorted_by_file_name["swept_value"]=f'{attribute_name}'
### The second entry in all_result saves the config file that is the same for all
all_results_sorted_by_file_name["parameters"]=params
all_results_sorted_by_file_name["n"]=n
all_results_sorted_by_file_name["m"]=m
all_results_sorted_by_file_name["k"]=k
all_results_sorted_by_file_name["measurment_type"]=mode

#getting file names
#sample_dict=all_results[f'{attribute_values[0]}']
file_names_as_keys=all_results[f'{attribute_values[0]}'].keys()
for key in file_names_as_keys:
    all_results_sorted_by_file_name[key]={}

for key in file_names_as_keys:
    for val in attribute_values:
      all_results_sorted_by_file_name[key][f'{val}']=all_results[f'{val}'][key]

# for file_name in os.listdir(folder_path):
#    for val in attribute_values:
#    for val in attribute_values:
#        val_dict=a




# Convert the dictionary before serializing
converted_dict = convert_numpy_types(all_results_sorted_by_file_name)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the dictionary to a JSON file
with open(f'{params["res_path"]}sweeping_{attribute_name}_{mode}_{current_time}.json', 'w') as json_file:
      # json_file.write('{\n')  # Start of JSON object
      # for key, value in all_results_sorted_by_file_name.items():
      #    json_file.write(f'   "{key}": {value},\n')  # Write each key-value pair
      # json_file.seek(json_file.tell() - 2, 0)  # Remove the last comma
      # json_file.write('\n}')  # End of JSON object
      json.dump(converted_dict, json_file, indent=4)

# with open(f'{params["res_path"]}sweeping_{attribute_name}.json', 'r') as f:
#    my_dict = json.load(f)


####### removing data #############
# Specify the directory path
directory = params["folder_path"]

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, '*'))

# Loop through and remove each file
for file in files:
    os.remove(file)

directory = params["truth_path"]

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, '*'))

# Loop through and remove each file
for file in files:
    os.remove(file)




