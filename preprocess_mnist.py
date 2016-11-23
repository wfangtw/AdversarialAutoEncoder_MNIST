import numpy as np

def get_data():
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='./')

    return mnist.data, mnist.target

def split(X, y):
    val_X = X[60000:].astype(np.float64)
    val_y = y[60000:].astype(np.float64)
    
    dat_X = X[:60000]
    dat_y = y[:60000]

    train_X = np.zeros((4000, dat_X.shape[1]))
    train_y = np.zeros(4000)

    idx = 0
    for lab in range(10):
        i = idx
        while i < 60000 and dat_y[i] == lab:
            i += 1
        cur_X = dat_X[idx:i].astype(np.float64)
        cur_y = dat_y[idx:i].astype(np.float64)
        perm = np.random.permutation(cur_y.shape[0])
        train_idx = perm[:400]
        unlabeled_idx = perm[400:]

        train_X[lab * 400 : (lab + 1) * 400] = cur_X[train_idx]
        train_y[lab * 400 : (lab + 1) * 400] = cur_y[train_idx]

        if lab == 0:
            unlabeled_X = cur_X[unlabeled_idx]
        else:
            unlabeled_X = np.vstack([unlabeled_X, cur_X[unlabeled_idx]])

        idx = i

    return train_X, train_y, val_X, val_y, unlabeled_X


if __name__ == '__main__':
    X, y = get_data()
    train_X, train_y, val_X, val_y, unlabeled_X = split(X, y)
    np.save('npy_mnist/train_X', train_X)
    np.save('npy_mnist/train_y', train_y)
    np.save('npy_mnist/val_X', val_X)
    np.save('npy_mnist/val_y', val_y)
    np.save('npy_mnist/unlabeled_X', unlabeled_X)
