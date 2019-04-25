import numpy as np


def batch_idx(i, size, N):
    if size >= N:
        return np.arange(N)
    idx1 = (i*size) % N
    idx2 = ((i+1)*size) % N
    if idx2 < idx1:
        idx1 -= N
    return np.arange(idx1, idx2)


def shuffle_data(X, Y):
    p = np.random.permutation(len(X))
    return X[p], Y[p]


def f_theta(theta, Siy, C, cover_indices):
    d = len(Siy)
    
    # calculate v
    v = np.zeros(d)
    for j in cover_indices:
        for i in range(d):
            if Siy[i] == 1 and C[i][j] == 1:
                v[j] = 1
                break

    # must cover itself in our setting
    for j in range(d):
        if Siy[j] == 1:
            v[j] = 1

    return np.dot(theta, v), v


def phi_theta(theta, Si, C, alphaz, cover_indices):
    res = []
    gradient = np.zeros_like(theta)
    for y, e in enumerate(alphaz):
        if e == 1:
            ft, v = f_theta(theta, Si[y], C, cover_indices)
            res.append(ft)
            gradient += v
    sum_alphaz = alphaz.sum()
    return sum(res)/sum_alphaz, gradient/sum_alphaz


def obj_gradient(theta, Ss, zs, C, alphazs, lam=1, mu=1, cover_indices=[]):
    res = 0
    N, m = len(Ss), len(Ss[0])
    gradient = np.zeros_like(theta)
    for i in range(N):
        # calculate hinge loss
        max_loss = 0
        grad_1 = None
        phi, grad_2 = phi_theta(theta, Ss[i], C, alphazs[i], cover_indices)
        for y in range(m):
            ft, v = f_theta(theta, Ss[i][y], C, cover_indices)
            loss = max((1 if alphazs[i][y] == 1 else 0) + ft - phi, 0)
            if loss > max_loss:
                max_loss = loss
                grad_1 = v
        if max_loss > 0:
            gradient += grad_1 - grad_2
    return (theta + lam * gradient/N - mu * np.ones_like(theta))/N


def sto_gradient_descent(theta, Ss, zs, C, alphazs, batch_size=10, eta=0.01, iters=1000, lam=1, mu=1, print_every=10, cover_indices=[]):
    N = len(Ss)
    for i in range(iters):
        idx = batch_idx(i, batch_size, N)
        gradient = obj_gradient(theta, Ss[idx], zs[idx], C, alphazs[idx], lam, mu, cover_indices)
        theta -= eta * gradient
        if print_every > 0 and (i+1) % print_every == 0:
            print('Iter %d, theta: %s, gradient: %s' % (i+1, theta, gradient))
    return theta


def dops(X, Y, T, C, m, alpha, init, loss='approx', batch_size=10, eta=0.01, iters=1000, lam=1, mu=1, print_every=10, cover_indices=[]):
    # config
    M, d = X.shape
    N = int(np.floor(M/m))
    X, Y = shuffle_data(X, Y)
    Ss = np.array([X[i*m:(i+1)*m] for i in range(N)])
    zs = np.array([Y[i*m:(i+1)*m] for i in range(N)])
    if loss == 'approx':
        maxzs = np.array([z.max() for z in zs])
        alphazs = np.array([np.array([1 if e >= maxz*alpha else 0 for e in z]) for z, maxz in zip(zs, maxzs)])
    elif loss == 'quantile':
        quantilezs = np.array([np.percentile(z, alpha*100) for z in zs])
        alphazs = np.array([np.array([1 if e >= quantilez else 0 for e in z]) for z, quantilez in zip(zs, quantilezs)])
    else:
        raise ValueError('loss function is None!')

    # optimize
    theta = sto_gradient_descent(init, Ss, zs, C, alphazs, batch_size, eta, iters, lam, mu, print_every,cover_indices)
    if theta is None:
        raise ValueError('theta is None!')

    # get result
    res = [f_theta(theta, t, C, cover_indices)[0] for t in T]
    return res, theta, np.argmax(res)
