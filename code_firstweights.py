# LOAD PACKAGES
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio           # importing matlab data
import matplotlib.pyplot as plt  # plotting
#%matplotlib inline
from copy import deepcopy

import sys
#import networkNIM.networkNIM as NIM
#import networkNIM.sinNIM as sinNIM
import NDN.NDNutils as NDNutils
import NDN.NDN as NDN

def downsample(X):
    import scipy.misc
    X = np.reshape(X,(X.shape[0],31,31))
    X = np.array([scipy.misc.imresize(Z,size=(15,15),interp='bicubic') for Z in X])
    #X = X[:,5:26,5:26]
    X = np.reshape(X,(X.shape[0],15*15))
    return X

# PREPARE DATA
stim1 = downsample(np.load('./Data/region1/training_inputs.npy'))
Robs1 = np.load('./Data/region1/training_set.npy')
stim2 = downsample(np.load('./Data/region2/training_inputs.npy'))
Robs2 = np.load('./Data/region2/training_set.npy')
stim3 = downsample(np.load('./Data/region3/training_inputs.npy'))
Robs3 = np.load('./Data/region3/training_set.npy')

# JM
print("stim1 - min and max: ",np.min(stim1),",",np.max(stim1))
print("Robs1 - min and max: ",np.min(Robs1),",",np.max(Robs1))
print("stim1.shape = ",stim1.shape[0],",",stim1.shape[1])
print("stim2.shape = ",stim2.shape[0],",",stim2.shape[1])
print("stim3.shape = ",stim3.shape[0],",",stim3.shape[1])
print("Robs1.shape = ",Robs1.shape[0],",",Robs1.shape[1])
print("Robs2.shape = ",Robs2.shape[0],",",Robs2.shape[1])
print("Robs3.shape = ",Robs3.shape[0],",",Robs3.shape[1])

# Normalize stim by overall values
stim_norm = np.mean([np.std(stim1),np.std(stim2),np.std(stim3)])
stim1=stim1/stim_norm
stim2=stim2/stim_norm
stim3=stim3/stim_norm
# Check inputs
NX=15
#plt.imshow(np.reshape(stim1[1,:],[NX,NX]),cmap='Greys',interpolation='none')
#plt.show()
NT,Ncells = Robs1.shape
print(NT,'frames, Ncells = ', Ncells)

# Assemble two different experiments
# 1: concatenate stim
NT1,NP1 = stim1.shape
NT2,NP2 = stim2.shape
NT3,NP3 = stim3.shape
NT = NT1+NT2+NT3
stim = np.concatenate((stim1,stim2,stim3),axis=0)
#NT,NX = stim.shape
print(NT,NX)
# Assemble responses in different places
NC1=Robs1.shape[1]
NC2=Robs2.shape[1]
NC3=Robs3.shape[1]
NC = NC1+NC2+NC3
Robs_full = np.zeros([NT,NC],dtype='float32')
val_resps = np.zeros([NT,NC],dtype='float32')
Robs_full[0:NT1,0:NC1]=Robs1
Robs_full[NT1:(NT1+NT2),NC1:(NC1+NC2)]=Robs2
Robs_full[(NT1+NT2):NT,(NC1+NC2):NC]=Robs3
val_resps[0:NT1,0:NC1]=1.0
val_resps[NT1:(NT1+NT2),NC1:(NC1+NC2)]=1.0
val_resps[(NT1+NT2):NT,(NC1+NC2):NC]=1.0

# Cross-validation set
XVstim1 = downsample(np.load('./Data/region1/validation_inputs.npy'))   
XVRobs1 = np.load('./Data/region1/validation_set.npy')
XVstim2 = downsample(np.load('./Data/region2/validation_inputs.npy'))
XVRobs2 = np.load('./Data/region2/validation_set.npy')
XVstim3 = downsample(np.load('./Data/region3/validation_inputs.npy'))
XVRobs3 = np.load('./Data/region3/validation_set.npy')
RobsXV = np.concatenate( (XVRobs1, XVRobs2, XVRobs3), axis=1)
NTXV=XVstim1.shape[0]

XVstim1=XVstim1/stim_norm
XVstim2=XVstim2/stim_norm
XVstim3=XVstim3/stim_norm


# I need to design eval_models to be able to take different size stims, but as-is now, it does
# not rebuild the graph, and thus can only cross-validate using indices within the existing stim
# So, for now, I will concatenate the cross-validation set with the normal

# Combined stimulus and Robs
NTfit = NT
NTXV = 150

Cstim = np.concatenate( (stim, XVstim1, XVstim2, XVstim3), axis=0)
Crobs = np.concatenate( (Robs_full, RobsXV, RobsXV, RobsXV), axis=0)  # note all Robs in all three (lazy)
Cval_resps = np.concatenate( (val_resps, np.ones([150,NC])), axis=0)
Ui=range(NT)
Xi=range(NT,NT+150)

# And individual stim-resp combos
Cstim1 = np.concatenate( (stim1, XVstim1), axis=0 )
Crobs1 = np.concatenate( (Robs1, XVRobs1), axis=0 )
Cstim2 = np.concatenate( (stim2, XVstim2), axis=0 )
Crobs2 = np.concatenate( (Robs2, XVRobs2), axis=0 )
Cstim3 = np.concatenate( (stim3, XVstim3), axis=0 )
Crobs3 = np.concatenate( (Robs3, XVRobs3), axis=0 )

def evaluate_performance(pred,XVRobs):
    import scipy.stats
    c = []

    for i in range(pred.shape[1]):
        c.append(scipy.stats.pearsonr(np.array(pred)[:,i].flatten(),XVRobs[:,i].flatten())[0])

    print("Number of NaN corrs: ",np.sum(np.nan_to_num(c)==0))
    return np.mean(np.nan_to_num(c))

hls = 40
d2x = 0.0005
l1 = 0.000001

HSMparams1 = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NX], layer_sizes=[hls,2*hls,NC1], ei_layers=[0,int(hls/2)], normalization=[0],
    layer_types=['normal','normal','normal'], reg_list={'d2x':[d2x,None,None],'l1':[l1,None,None],'max':[None,None,100]})
HSM1 = NDN.NDN( HSMparams1, noise_dist='gaussian' )

# Display randomly generated weights in first 8 neurons in 1st layer
fig, ax = plt.subplots(nrows=3, ncols=4)
fig.set_size_inches(12, 6)
weights = np.array([])
for nn in range(8):
    plt.subplot(2, 4, nn+1)
    k = HSM1.networks[0].layers[0].weights[:,nn]
    plt.imshow(np.reshape(k,[NX,NX]),cmap='Greys',interpolation='none', vmin=-max(abs(k)),vmax=max(abs(k)))
    print("Neuron ",nn," weights - min and max: ",min(k),", ",max(k))
    weights = np.concatenate((weights, k))
plt.show()
# Display the distribution of weights
print(np.shape(weights))
nn=0
import matplotlib as mlab
n, bins, patches = plt.hist(weights, 20, facecolor='green')
plt.show()