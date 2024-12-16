import openTSNE
import umap
import hnne
import cne
import trimap
import phate
from sklearn.decomposition import PCA
from pacmap import LocalMAP, PaCMAP
from data import data_prep
import argparse
import time
import os
import tqdm

parser = argparse.ArgumentParser(description='Run DR experiments')
parser.add_argument('--data', type=str, default='MNIST', help='Dataset')
parser.add_argument('--method', type=str, default='pacmap', help='Method')
parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
args = parser.parse_args()

X, y = data_prep(args.data)

for i in tqdm.tqdm(range(args.iterations)):
    # check if the combination and the method already exists
    # if os.path.exists(f'embedding/{args.data}_{args.method}_{i}.npy'):
    #     continue

    if args.method == 'pacmap':
        import pacmap
        model = pacmap.PaCMAP()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "localmap":
        from LocalMAP import LocalMAP
        model = LocalMAP()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "TSNE":
        # import openTSNE
        # model = openTSNE.TSNE(n_jobs=-1)
        from sklearn.manifold import TSNE
        model = TSNE()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "UMAP":
        import umap
        model = umap.UMAP()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "HNNE":
        import hnne
        model = hnne.HNNE()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "InfoNCE":
        import cne
        model = cne.CNE()
        start_time = time.time()
        X_trans = model.fit_transform(X.astype(float))
        total_time = time.time()-start_time
    elif args.method == "NegTSNE":
        import cne
        model = cne.CNE(loss_mode="neg")
        start_time = time.time()
        X_trans = model.fit_transform(X.astype(float))
        total_time = time.time()-start_time
    elif args.method == "NCVi":
        import cne
        model = cne.CNE(loss_mode="nce",optimizer="adam",parametric=True)
        start_time = time.time()
        X_trans = model.fit_transform(X.astype(float))
        total_time = time.time()-start_time
    elif args.method == "TriMAP":
        import trimap
        model = trimap.TRIMAP()
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "PHATE":
        import phate
        model = phate.PHATE(n_jobs=-1)
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    elif args.method == "PCA":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2)
        start_time = time.time()
        X_trans = model.fit_transform(X)
        total_time = time.time()-start_time
    import numpy as np
    # save the embedding into the embedding folder
    np.save(f'embedding/{args.data}_{args.method}_{i}.npy',X_trans)
    
    # plot the X_trans
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,7))
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, s=0.1, cmap="Spectral")
    plt.savefig(f'visualization/{args.data}_{args.method}_{i}.png')
    plt.clf()
    plt.close()

    # save the running time information into the csv file
    if os.path.exists('running_time.csv'):
        with open('running_time.csv','a') as f:
            f.write(f'{args.data},{args.method},{i},{total_time}\n')
    else:
        with open('running_time.csv','w') as f:
            f.write('data,method,iteration,time\n')
            f.write(f'{args.data},{args.method},{i},{total_time}\n')
