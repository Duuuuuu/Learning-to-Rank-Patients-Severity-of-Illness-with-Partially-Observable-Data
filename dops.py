from autograd import numpy as np
from autograd import grad


def shuffle_data(X, Y):
	p = np.random.permutation(len(X))
	return X[p], Y[p]


def f_theta(theta, Siy, C):
	d = len(S)
	CS = []
	v = np.zeros(d)

	# calculate v
	for j in range(d):
		for i in range(d):
			if Siy[i] == 1 and j in C[i]:
				v[j] = 1
				break

	return np.dot(theta, v)


def phi_theta(theta, Si, C, alphaz):
	return sum([f_theta(theta, Si[y], C) for y in alphaz])/len(alphaz)


def objective(theta, Ss, zs, C, alphazs):
	res = 0
	N, m = len(Ss), len(Ss[0])
	for i in range(N):
		# calculate hinge loss
		hinge_losses = np.zeros(m)
		for y in range(m):
			hinge_losses[y] = max((1 if y in alphazs[i] else 0) + f_theta(theta, Ss[i][y]) \
							  - phi_theta(theta, Ss[i], zs[i], C), 0)
		res += hinge_losses.max()
	return res


def gradient_descent(theta, Ss, zs, C, alphazs, eta=0.05, iters=1000, verbose=False):
	gradident_fun = grad(objective)
	for i in range(iters):
		if verbose:
			print(theta)

		# gd
		theta -= eta * gradident_fun(theta, Ss, zs, C, alphazs)

	return theta


def dops(X, Y, T, C, m, alpha, eta=0.05, iters=1000):
	# config
	M, d = X.shape
	N = np.floor(M/m)
	X, Y = shuffle_data(X, Y)
	Ss = [X[i*m:(i+1)*m] for i in range(N)]
	zs = [Y[i*m:(i+1)*m] for i in range(N)]
	maxzs = [e.max() for e in zs] 
	alphazs = [[i if e >= maxz/alpha for i, e in enumerate(z)] for z, maxz in zip(zs, maxzs)]

	# optimize
	# init theta = 0
	theta = gradient_descent(np.zeros(d), Ss, zs, C, alphazs, eta, iters)
	
	# get result
	return max([f_theta(theta, t, C) for t in T])
