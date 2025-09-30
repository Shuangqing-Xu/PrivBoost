import pandas as pd
import numpy as np
import random

def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def scale(y):
    max_scale = 1
    min_scale = -1
    return (max_scale-min_scale)*(y-y.min())/(y.max()-y.min()) + min_scale

def distributed_scale(client_y, y_max, y_min):
    max_scale = 1
    min_scale = -1
    return (max_scale-min_scale)*(client_y-y_min)/(y_max-y_min) + min_scale

def rescale(y, y_test, y_pred):
    # mean = np.full((y_test.shape[0],), y.mean().values[0], dtype=float)
    # mean2 = np.full((y_test.shape[0],1), y.mean().values[0], dtype=float)

    max_scale = 1
    min_scale = -1

    max = y.max().values[0]
    min = y.min().values[0]
    #y = (y_scale-min_scale)*(max-min)/(max_scale-min_scale)+min
    y_pred = (y_pred-min_scale)*(max-min)/(max_scale-min_scale)+min
    y_test = (y_test-min_scale)*(max-min)/(max_scale-min_scale)+min
    return y_pred, y_test

def rescale_rmse(y, rmse):
    max_scale = 1
    min_scale = -1

    max = y.max().values[0]
    min = y.min().values[0]
    #y = (y_scale-min_scale)*(max-min)/(max_scale-min_scale)+min
    rmse = (rmse-min_scale)*(max-min)/(max_scale-min_scale)+min
    return rmse

def horizontally_split_data(n, X_train, X_test, y_train, y_test):
    client_X_train, client_X_test, client_y_train, client_y_test = [], [], [], []

    num = 0
    for i in range(0,n):
        start = num
        num = num + int(X_train.shape[0] / n)
        if i == n-1:
            num = X_train.shape[0]
        client_X_train.append(X_train.iloc[start:num,])
        client_y_train.append(y_train.iloc[start:num,])

    # num = 0
    # for i in range(0,n):
    #     start = num
    #     num = num + int(X_test.shape[0] / n)
    #     if i == n-1:
    #         num = X_test.shape[0]
    #     client_X_test.append(X_test.iloc[start:num,])
    #     client_y_test.append(y_test.iloc[start:num,])

    return client_X_train, client_X_test, client_y_train, client_y_test

def non_iid_split_data_wine(n, X_train, y_train):
    X = pd.concat([X_train,y_train],axis=1)
    dataset = X
    class_id = 11

    dataset = np.array(dataset)
    num_class = int(np.max(dataset[:,class_id])) + 1

    net_dataidx_map = partition_data(dataset, class_id, num_class, "noniid-labeldir", 5, 0.5, 1234)

    client_X_train, client_y_train = [], []
    client_X_y_train = []
    for i in range(n):
        client_X_y_train.append(dataset[net_dataidx_map[i]])
        # client_y_train.append(dataset[net_dataidx_map[i]])

    return client_X_y_train

def iid_split_data_wine(n, X_train, y_train):
    X = pd.concat([X_train,y_train],axis=1)
    dataset = X
    class_id = 11

    dataset = np.array(dataset)

    num = 0
    client_X_train, client_y_train = [], []
    client_X_y_train = []
    for i in range(n):
        start = num
        num = num + int(X_train.shape[0] / n)
        if i == n-1:
            num = X_train.shape[0]
        client_X_y_train.append(dataset[start:num,])

    return client_X_y_train


def horizontally_split_data_numpy(n, X_train, X_test, y_train, y_test):
    client_X_train, client_X_test, client_y_train, client_y_test = [], [], [], []

    num = 0
    for i in range(0,n):
        start = num
        num = num + int(X_train.shape[0] / n)
        if i == n-1:
            num = X_train.shape[0]
        client_X_train.append(X_train[start:num,])
        client_y_train.append(y_train[start:num,])

    num = 0
    for i in range(0,n):
        start = num
        num = num + int(X_test.shape[0] / n)
        if i == n-1:
            num = X_test.shape[0]
        client_X_test.append(X_test[start:num,])
        client_y_test.append(y_test[start:num,])

    return client_X_train, client_X_test, client_y_train, client_y_test


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def partition_data(dataset, class_id, K, partition, n_parties, beta, seed):
    np.random.seed(seed)
    random.seed(seed)

    n_train = dataset.shape[0]
    y_train = dataset[:,class_id]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10

        N = dataset.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        
        times=[0 for i in range(K)]
        contain=[]
        for i in range(n_parties):
            current=[i%K]
            times[i%K]+=1
            j=1
            while (j<num):
                ind=random.randint(0,K-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
        for i in range(n_parties):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

        for i in range(n_parties):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    return net_dataidx_map

def abalone():
    dataset = pd.read_csv("./data/abalone.data")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[0], axis=1)
    return X, y

def cal_housing():
    dataset = pd.read_csv("./data/cal_housing.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[4], axis=1)
    return X, y

def energy():
    dataset = pd.read_csv("./data/energy.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def bos_housing():
    dataset = pd.read_csv("./data/bos_housing.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def wine():
    dataset = pd.read_csv("./data/wine.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def CASP():
    dataset = pd.read_csv("./data/CASP.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def plant():
    dataset = pd.read_csv("./data/plant.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def bank():
    dataset = pd.read_excel("./data/plant.xlsx")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def air():
    dataset = pd.read_csv("./data/air.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[0], axis=1)
    X=X.drop(X.columns[0], axis=1)
    return X, y

def bike():
    dataset = pd.read_csv("./data/bike.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[0], axis=1)
    X=X.drop(X.columns[0], axis=1)
    return X, y

def pol():
    dataset = pd.read_csv("./data/pol.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def elevators():
    dataset = pd.read_csv("./data/elevators.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    return X, y

def housing():
    dataset = pd.read_csv("./data/housing.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[4], axis=1)
    return X, y

def blog():
    dataset = pd.read_csv("./data/blog.csv")
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1:]
    X=X.drop(X.columns[4], axis=1)
    return X, y