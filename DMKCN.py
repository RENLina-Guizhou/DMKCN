from __future__ import print_function, division
import argparse
import random
import sqlite3
from multiprocessing import Pool

import numpy as np
import sklearn
from scipy.sparse import csgraph

from sklearn import cluster
from sklearn.cluster import KMeans,spectral_clustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, pairwise_kernels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear


from compute_kernels import compute_kernels, compute_kernelst, compute_kernelsp,  compute_kernelsg

from utils import load_data

from evaluation import eva
import math
from collections import Counter
# from sklearn.decomposition import TruncatedSVD
# from sklearn.random_projection import sparse_random_matrix

class AE(nn.Module):
    # 初始化
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    # 反向传播
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

class KAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,args):
        super(KAE, self).__init__()
        n_input=args.n_input
        n_z=args.n_z
        data_num=args.data_num
        n_clusters = args.n_clusters
        self.args=args
        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.k1=Linear(data_num,2000)
        self.k2=Linear(2000,500)
        self.k3=Linear(500,500)
        self.k4=Linear(500,n_input)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(args.n_clusters, data_num))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # cluster layer
        self.cluster_layer1 = Parameter(torch.Tensor(args.n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.b = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.c =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.d = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.e = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.f =  nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.g = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.h = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.e1 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.f1 = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.g1 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.h1 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.e2 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.f2 = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.g2 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.h2 = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.e3 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.f3 =  nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.g3 =nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)
        self.h3 = nn.Parameter(nn.init.constant_(torch.zeros(data_num, data_num), 1), requires_grad=True).to(device)

    def forward(self, x):

        enc_h1 = self.ae.enc_1(x)

        k1h1 = compute_kernels(enc_h1)
        kzh1 = compute_kernelst(enc_h1)
        kph1 = compute_kernelsp(enc_h1)
        kg1 = compute_kernelsg(enc_h1.detach().cpu().numpy() )
        kg1=torch.tensor(kg1)
        kh1 = self.e1*k1h1.to(device) + self.f1*kzh1.to(device) + self.g1*kph1.to(device)+self.h1*kg1.to(device)


        enc_h2 = self.ae.enc_2(enc_h1)

        k1h2 = compute_kernels(enc_h2)
        kzh2 = compute_kernelst(enc_h2)
        kph2 = compute_kernelsp(enc_h2)
        kg2 = compute_kernelsg(enc_h2.detach().cpu().numpy() )
        kg2=torch.tensor(kg2)
        kh2 = self.e2*k1h2.to(device) + self.f2*kzh2.to(device) + self.g2*kph2.to(device)+self.h2*kg2.to(device)
        enc_h3 = self.ae.enc_3(enc_h2)


        k1h3 = compute_kernels(enc_h3)
        kzh3 = compute_kernelst(enc_h3)
        kph3 = compute_kernelsp(enc_h3)
        kg3 = compute_kernelsg(enc_h3.detach().cpu().numpy() )
        kg3=torch.tensor(kg3)
        kh3 = self.e3*k1h3.to(device) + self.f3*kzh3.to(device) + self.g3*kph3.to(device)+self.h3*kg3.to(device)
        z = self.ae.z_layer(enc_h3)


        k1 = compute_kernels(z)
        kz = compute_kernelst(z)
        kp = compute_kernelsp(z)
        kgz = compute_kernelsg(z.detach().cpu().numpy() )
        kgz=torch.tensor(kgz)
        kz = self.e*k1.to(device) + self.f*kz.to(device) + self.g*kp.to(device)+self.h*kgz.to(device)
        k=self.a*kh1+self.b*kh2+self.c*kh3+self.d*kz

        k1=self.k1(1*k)
        k22=self.k2(1*k1)
        k33=self.k3(1*k22)
        x_bar=self.k4(1*k33)

        q = k.unsqueeze(1) - self.cluster_layer



        return x_bar,k,z,enc_h1,enc_h2,enc_h3,q,k22,k33,kh1,kh2,kh3,kz

def kernel_kmeans(Kt, n_clusters=2, max_iter=100, n_init=5, tol=1e-5):
    U, _, _ = np.linalg.svd(Kt)
    H = U[:,0:n_clusters]
    from sklearn.cluster import KMeans
    km1 = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, tol=tol).fit(H)
    mem = km1.labels_
    dist = np.zeros((Kt.shape[0], n_clusters))
    diagKt = np.diag(Kt)
    for j in range(n_clusters):
        if (mem==j).sum() > 1:
            dist[:,j] = (diagKt
                - (2 * Kt[:,mem==j].sum(axis=1) / (mem==j).sum())
                + (Kt[mem==j,:][:,mem==j].sum() / ((mem==j).sum()**2)))
    return mem, dist
def train_kae(runtime,dataset, datasetname, device):
    model = KAE(500, 500, 2000, 2000, 500, 500, args=args).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        x_bar,k,z,enc_h1,enc_h2,enc_h3,q,k22,k33,kh1,kh2,kh3,kz= model(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(k.detach().cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(100):
        if epoch % 1 == 0:
            # update_interval
            x_bar,k,z,enc_h1,enc_h2,enc_h3,q,k22,k33,kh1,kh2,kh3,kz= model(data)

            y_pred,dist = kernel_kmeans(k.detach().cpu().numpy(), args.n_clusters)
            eva(y, y_pred, epoch,runtime, datasetname)

        loss = 1 * F.mse_loss(x_bar, data) + 1 * torch.sum(q)  \
               + 1 * F.mse_loss(enc_h1, torch.matmul(k, enc_h1)) + 1 * F.mse_loss(enc_h2, torch.matmul(k, enc_h2)) \
               + 1 * F.mse_loss(enc_h3, torch.matmul(k, enc_h3)) + 1 * F.mse_loss(z, torch.matmul(k, z))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    con = sqlite3.connect("result.db")
    cur = con.cursor()
    cur.execute("delete from sdcn ")
    con.commit()

    datasets = ['bbcsport']
    for dataset in datasets:
        batch =10 # 运行轮次
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default=dataset)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        #parser.add_argument('--data_num', default=20000, type=int)
        #parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)


        if args.name == 'bbc':
            args.k =3
            args.n_clusters = 5
            args.n_input = 9635
            args.data_num=2225

        if args.name=='bbcsport':
            args.n_clusters = 5
            args.n_input = 4613
            args.data_num = 737

        print(args)
        for i in range(batch):
            cur.execute("delete from sdcn where datasetname=? and batch=?", [args.name, i])
            con.commit()
            train_kae(i, dataset, args.name, device)

    _datasets = ['bbc','bbcsport']
    for name in _datasets:
        datas = cur.execute(
            "select datasetname,batch,epoch,acc,nmi,ari,f1 from sdcn where datasetname=? order by batch",
            [name]).fetchall()
        for d in datas:
            if d is not None:
                print('dataname:{0},batch:{1},epoch:{2}'.format(d[0], d[1], d[2]), 'acc {:.4f}'.format(d[3]),
                      ', nmi {:.4f}'.format(d[4]), ', ari {:.4f}'.format(d[5]),
                      ', f1 {:.4f}'.format(d[6]))
    for name in _datasets:
        result = cur.execute(
            "select  avg(acc) as acc,avg(nmi) as nmi,avg(ari) as ari,avg(f1) as f1 from ( select acc,nmi,ari,f1 from sdcn where datasetname =? order by nmi desc limit 10)",
            [name]).fetchone()

        if result[0] is not None:
            print('dataname:{0}'.format(name), 'AVG :acc {:.4f}'.format(result[0]),
                  ', nmi {:.4f}'.format(result[1]), ', ari {:.4f}'.format(result[2]),
                  ', f1 {:.4f}'.format(result[3]))
    cur.close()
    con.close()
