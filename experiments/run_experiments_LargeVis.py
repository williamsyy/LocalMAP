# To run LargeVis, you need to install the LargeVis package within python 2 environment.
# You need to create a new python 2 environment and install the LargeVis package.
# Please follow the instructions in the LargeVis repository: https://github.com/lferry007/LargeVis
import LargeVis
import data as data

from data import data_prep
import argparse
import time
import os
import tqdm

parser = argparse.ArgumentParser(description='Run DR experiments')
parser.add_argument('--data', type=str, default='neurips2021_total', help='Dataset')
parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
args = parser.parse_args()

X, y = data_prep(args.data)
# save the dataset into the format for LargeVis
# Get the dimensions of the array
rows, cols = X.shape

with open("txt_data/%s_data.txt"%args.data, 'w') as txt_file:
    # Write the dimensions to the text file
    txt_file.write(str(rows) + ' ' + str(cols) + '\n')

    # Write the array data to the text file
    for row in X:
        row_str = ' '.join(map(str, row))
        txt_file.write(row_str + '\n')
txt_file.close()

with open("txt_data/%s_label.txt"%args.data, 'w') as txt_file:
    for label in y:
        txt_file.write(str(label) + '\n')
txt_file.close()

for i in tqdm.tqdm(range(args.iterations)):
    # check if the combination and the method already exists
    if os.path.exists('embedding/%s_largevis_%d.npy'%(args.data, i)):
        continue
    import numpy as np
    LargeVis.loadfile("txt_data/%s_data.txt"%args.data)
    start_time = time.time()
    X_trans = LargeVis.run(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    total_time = time.time()-start_time
    np.save('embedding/%s_largevis_%d.npy'%(args.data, i), X_trans)
    
    # save the running time information into the csv file
    if os.path.exists('running_time.csv'):
        with open('running_time.csv','a') as f:
            f.write('%s,%s,%d,%f\n'%(args.data,"largevis",i,total_time))
    else:
        with open('running_time.csv','w') as f:
            f.write('data,method,iteration,time\n')
            f.write('%s,%s,%d,%f\n'%(args.data,"largevis",i,total_time))