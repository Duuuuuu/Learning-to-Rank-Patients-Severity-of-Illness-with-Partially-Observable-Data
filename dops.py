# from autograd import numpy as np
# from autograd import grad
from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np


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

	return np.dot(theta, v), v


def phi_theta(theta, Si, C, alphaz):
	res = []
	gradient = np.zeros_like(theta)
	for y, e in enumerate(alphaz):
		if e == 1:
			ft, v = f_theta(theta, Si[y], C)
			res.append(ft)
			gradient += v
	return sum(res)/alphaz.sum(), gradient/alphaz.sum()


def obj_gradient(theta, Ss, zs, C, alphazs, lam=1, mu=1):
	res = 0
	N, m = len(Ss), len(Ss[0])
	gradient = np.zeros_like(theta)
	for i in range(N):
		# calculate hinge loss
		max_loss = 0
		grad_1 = None
		phi, grad_2 = phi_theta(theta, Ss[i], C, alphazs[i])
		for y in range(m):
			ft, v = f_theta(theta, Ss[i][y], C)
			loss = max((1 if alphazs[i][y] == 1 else 0) + ft - phi, 0)
			if loss > max_loss:
				max_loss = loss
				grad_1 = v
		if max_loss > 0:
			gradient += grad_1 - grad_2
	return theta + lam * gradient/N - mu * np.ones_like(theta)


def sub_gradient_descent(theta, Ss, zs, C, alphazs, maxzs, eta=0.01, iters=1000, lam=1, mu=1, print_every=10):
	best_score = 0
	best_theta = None
	for i in range(iters):
		score = 0
		gradient = obj_gradient(theta, Ss, zs, C, alphazs, lam, mu)
		theta -= eta * gradient
		if print_every > 0 and (i+1) % print_every == 0:
			print('Iter %d, theta: %s, gradient: %s' % (i+1, theta, gradient))

		# select the best one, but do we need this?
		# for S, maxz in zip(Ss, maxzs):
		# 	if max([f_theta(theta, s, C)[0] for s in S]) == maxz:
		# 		score += 1
		# if score > best_score:
		# 	best_score = score
		# 	best_theta = theta

	return theta


def dops(X, Y, T, C, m, alpha, init, eta=0.01, iters=1000, lam=1, mu=1, print_every=10):
	# config
	M, d = X.shape
	N = int(np.floor(M/m))
	X, Y = shuffle_data(X, Y)
	Ss = [X[i*m:(i+1)*m] for i in range(N)]
	zs = [Y[i*m:(i+1)*m] for i in range(N)]
	maxzs = [e.max() for e in zs]
	alphazs = [np.array([1 if e >= maxz*alpha else 0 for e in z]) for z, maxz in zip(zs, maxzs)]

	# optimize
	theta = sub_gradient_descent(init, Ss, zs, C, alphazs, maxzs, eta, iters, lam, mu, print_every)
	if theta is None:
		raise ValueError('theta is None!')

	# get result
	res = [f_theta(theta, t, C)[0] for t in T]
	return res, theta, np.argmax(res)
