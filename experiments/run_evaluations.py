import numpy as np
import pandas as pd
from run_eval import knn_eval, svm_eval
import os
import data as data
import tqdm
# import silhouette score
from sklearn.metrics import silhouette_score
import argparse

# create the parser
parser = argparse.ArgumentParser(description='Run DR experiments')
# add the arguments
parser.add_argument('--data', type=str, default='MNIST', help='Dataset')
parser.add_argument('--method', type=str, default='UMAP', help='Method')

# load the evaluation.csv
df = pd.read_csv('evaluation.csv')

# parse the arguments
args = parser.parse_args()

for data_name in [args.data]:
    X, y = data.data_prep(data_name)
    for method in [args.method]:
        for i in os.listdir(f"embedding"):
            if i.startswith(f"{data_name}_{method}"):
                print("Starting to deal with",i)
                index = i[:-4].split("_")[-1]
                print(f"{data_name}_{method}_{index}:\n")
                if len(df[(df['data_name'] == data_name) & (df['method'] == method) & (df['embedding'] == int(index))]) > 0:
                    print(f"{data_name}_{method}_{index} has been evaluated\n")
                    continue

                # load the dataset
                embedding = np.load(f"embedding/{i}")
                # calculate the silhouette score
                silhou_score = silhouette_score(embedding, y)
                # calculate the knn score
                knn_score = knn_eval(embedding, y)
                # calculate the svm score
                svm_score = svm_eval(embedding, y)
                print(f"silhouette_score: {silhou_score}\n")
                print(f"knn_score: {knn_score}\n")
                print(f"svm_score: {svm_score}\n")
                # save the result into the csv file
                if os.path.exists('evaluation.csv'):
                    with open('evaluation.csv','a') as f:
                        f.write(f"{data_name},{method},{index},{silhou_score},{knn_score},{svm_score}\n")
                else:
                    with open('evaluation.csv','w') as f:
                        f.write("data_name,method,embedding,silhouette_score,knn_score,svm_score\n")
                        f.write(f"{data_name},{method},{index},{silhou_score},{knn_score},{svm_score}\n")