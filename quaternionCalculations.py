from __future__ import print_function

import sys
import os
import time
import math

import numpy as np
import theano
import theano.tensor as T

def quatDistance(q1,q2):
	return 1 - math.pow(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3],2)

def normQuat(q):
	return math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

def unitQuat(q):
	return [q[0]/normQuat(q),q[1]/normQuat(q),q[2]/normQuat(q),q[3]/normQuat(q)]

a = T.fmatrix()
b  = T.ftensor3()
w = T.reshape(a,(2,2,4))
#b = T.inv(w.norm(L=2,axis=2))
#c = T.nlinalg.AllocDiag()(b)
#k = T.dot(c,w)
f = theano.function([a],c)

def getLoss(target):
	x = target.size
	return x

print(f(np.array([  [-1,-1,1,1,1,1,1,1],[.2,.3,.2,.1,.5,.6,.4,.9]   ]).astype("float32")))
