from autograd import numpy as np
from autograd import grad


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


def f_theta(theta, Siy, C):
	d = len(Siy)

	# calculate v
	v = np.zeros(d)
	for j in range(d): # u
		for i in range(d): # i
			if Siy[i] == 1 and C[i][j] == 1:
				v[j] = 1
				break

	return np.dot(theta, v)


def phi_theta(theta, Si, C, alphaz):
	return sum([f_theta(theta, Si[y], C) for y, e in enumerate(alphaz) if e == 1])/alphaz.sum()


def objective(theta, Ss, zs, C, alphazs, lam=1, mu=1):
	res = 0
	N, m = len(Ss), len(Ss[0])
	for i in range(N):
		# calculate hinge loss
		max_loss = 0
		phi = phi_theta(theta, Ss[i], C, alphazs[i])
		for y in range(m):
			max_loss = max(max((1 if alphazs[i][y] == 1 else 0) + \
				f_theta(theta, Ss[i][y], C) - phi, 0), max_loss)
		res += max_loss
	return 0.5 * np.dot(theta, theta) + lam * res/N - mu * theta.sum()


def gradient_descent(theta, Ss, zs, C, alphazs, batch_size=10, eta=0.01, iters=1000, lam=1, mu=1, print_every=10):
	gradident_fun = grad(objective)
	N = len(Ss)
	for i in range(iters):
		idx = batch_idx(i, batch_size, N)
		theta -= eta * gradident_fun(theta, Ss[idx], zs[idx], C, alphazs, lam, mu)
		if print_every > 0 and (i+1) % print_every == 0:
			print('Iter %d : %s' % (i+1, theta))
	return theta


def dops(X, Y, T, C, m, alpha, init, batch_size=10, eta=0.01, iters=1000, lam=1, mu=1, print_every=10):
	# config
	M, d = X.shape
	N = int(np.floor(M/m))
	X, Y = shuffle_data(X, Y)
	Ss = np.array([X[i*m:(i+1)*m] for i in range(N)])
	zs = np.array([Y[i*m:(i+1)*m] for i in range(N)])
	maxzs = [e.max() for e in zs]
	alphazs = [np.array([1 if e >= maxz*alpha else 0 for e in z]) for z, maxz in zip(zs, maxzs)]

	# optimize
	theta = gradient_descent(init, Ss, zs, C, alphazs, batch_size, eta, iters, lam, mu, print_every)

	# get result
	res = [f_theta(theta, t, C) for t in T]
	return res, theta, np.argmax(res)
