#!/usr/bin/env python3

import math
import numpy as np
import operator
import os
import scipy.optimize
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class Args:
    pass


class Grad:

    def __init__(self, u, t, x):
        self.t = t
        self.x = x
        self.u = u

    def __call__(self, d):
        x, t = d
        ans = self.u
        for i in range(x):
            ans, = tf.gradients(ans, self.x)
        for i in range(t):
            ans, = tf.gradients(ans, self.t)
        return ans


class Func:

    def eval(self, mod, grad):
        u = grad((0, 0))
        ut = grad((0, 1))
        uxx = grad((2, 0))
        return ut - args.D * uxx


@tf.function
def func0(weights):
    u = neural_net(tf.stack((args.x_f, args.t_f), 1), weights)
    u_rect = neural_net(XT_rect, weights)
    f = args.func.eval(tf, Grad(u, args.t_f, args.x_f))
    loss = tf.reduce_mean(tf.square(args.u_rect - u_rect)) + tf.reduce_mean(
        tf.square(f))
    grads = tf.gradients(loss, weights)
    return loss, grads


def func(x):
    for w, lo, hi in zip(args.weights, offsets[:-1], offsets[1:]):
        w.flat = x[lo:hi]
    loss, grads = func0(args.weights)
    for g, lo, hi in zip(grads, offsets[:-1], offsets[1:]):
        args.grads[lo:hi] = g.numpy().flat
    sys.stderr.write("%g\n" % loss.numpy())
    return loss, args.grads


def latin_hypercube(n, samples):
    cut = np.linspace(0, 1, samples + 1)
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u, dtype=dtype)
    for j in range(n):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rdpoints, dtype=dtype)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]
    return H


def initial_weights(layers):
    weights = []
    for i in range(len(layers) - 1):
        scale = 1 / math.sqrt(layers[i])
        shape = layers[i], layers[i + 1]
        tmp = np.random.uniform(-scale, scale, shape)
        W = np.asarray(tmp, dtype=dtype)
        b = np.zeros(layers[i + 1], dtype)
        weights.append(W)
        weights.append(b)
    return weights


def neural_net(X, weights):
    n = len(weights) // 2
    for i in range(n):
        X = tf.tensordot(X, weights[2 * i], 1) + weights[2 * i - 1]
        if i == n - 1:
            break
        X = tf.tanh(X)
    return X[:, 0]


def exact(t, x):
    return np.exp(-args.D * (2 * math.pi)**2 * t) * np.sin(x * 2 * math.pi)


args = Args()
rect = Args()
dtype = np.float64
args.epochs = 10
args.Nu = 4096
args.Nb = 100
args.Nf = 10000
args.D = 0.05
args.layers = [2, 10, 10, 1]
t0 = 0
t1 = 1
x0 = -1
x1 = 1
rect.x0 = x0 + (x1 - x0) / 3
rect.x1 = x0 + (x1 - x0) * 2 / 3
rect.t0 = t0 + (t1 - t0) / 3
rect.t1 = t0 + (t1 - t0) * 2 / 3

Rect = latin_hypercube(2, args.Nu)
Rect *= rect.x1 - rect.x0, t1 - t0
Rect += rect.x0, rect.t0

tb = np.linspace(t0, t1, args.Nb)
tl = np.array([[x0, t] for t in tb], dtype=dtype)
tu = np.array([[x1, t] for t in tb], dtype=dtype)
XT_rect = np.vstack((Rect, tl, tu))

XT = latin_hypercube(2, args.Nf)
XT *= x1 - x0, t1 - t0
XT += x0, t0
args.x_f = tf.constant(XT[:, 0])
args.t_f = tf.constant(XT[:, 1])
args.u_rect = exact(XT_rect[:, 1], XT_rect[:, 0])

args.weights = initial_weights(args.layers)
x0 = np.concatenate([w.flat for w in args.weights])
offsets = [0]
cnt = 0
for w in args.weights:
    cnt += w.size
    offsets.append(cnt)
args.grads = np.empty(cnt, np.float64)

args.func = Func()
scipy.optimize.fmin_l_bfgs_b(func, x0, None, maxiter=args.epochs)
