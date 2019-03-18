#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:13 2019

@author: Minh Nguyen
"""

from graphviz import Digraph
from matplotlib import pyplot as plt
from math import log2
import numpy as np

def Gridify(string):
    size = int(len(string)**0.5)
    str_split = [ s for s in string ]
    str_list = [ str_split[i:i+size] for i in range(0, len(str_split),size) ]
    str_list = np.array(str_list, int)
    return str_list


'''
file_name = 'files/converge_states.txt'
sweeps = range(20)

fig = plt.figure()
plt.title('Converging energy of a Hopfield network')
plt.xlabel('Number of sweeps')
plt.ylabel('Energy')
plt.xticks(sweeps)

with open(file_name, 'r') as f:
    for line in f:
        line = line.strip().split()
        E = np.array(line, dtype=float)
        plt.plot(sweeps, E)

plt.show()
#plt.savefig('img/converging_energies.png', dpi=300)
'''


'''    REMEMBERING 'face' and 'tree'
img_name    = 'tree'
corr        = 'rand'
file_name = f'files/remembered_{img_name}.txt'
states = []
with open(file_name, 'r') as f:
    for line in f:
        line = line.strip().split()
        step = int(line[0])
        s = Gridify(line[1])
        states.append(s)
        fig = plt.figure()
        plt.title(f'Step {step}')
        plt.imshow(s)
        plt.savefig(f'img/snapshots/{img_name}_corr_{corr}_step_{step}.png', dpi=300)
'''


'''    HAMMING DISTANCE    
hamming_file = 'files/hamming.txt'
shape = (64,100)
hamming = np.zeros(shape, dtype=float)
with open(hamming_file, 'r') as hamm:
    for line in hamm:
        line = line.strip().split()
        k, p, H = int(line[0]), int(line[1]), float(line[2])
        hamming[k-1][p-1] = H

fig = plt.figure()
plt.title('Hamming Distance')
plt.imshow(hamming, origin='upper')
plt.gca().xaxis.tick_top()
plt.colorbar()
plt.savefig('img/hamming.png', dpi=300)
'''


a=np.loadtxt("files/landscape_energy.txt")
fig = plt.figure()
myDict = np.zeros(len(a))
plt.plot(a[:,0], a[:,1], 'o')
for i in range(0, len(a[:,0])):
    myDict[int(a[i,0])]=a[i,1]


b=np.loadtxt("files/landscape_connections.txt")
for i in range(0, len(b[:,0])):
    before=int(b[i,0])
    after=int(b[i,1])
    plt.plot([before,after], [myDict[before],myDict[after]], '--')

#plt.show()
plt.savefig('img/landscape_pyplot.png', dpi=300)





    
    
    
    
    
    
