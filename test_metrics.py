#!/usr/bin/python
"""
Test script for ICC(2,1) versus an implementation of the icc method implemented originally by Travis Smith in MATLAB from
Smith TB, Smith N. Agreement and reliability statistics for shapes. PLoS One. 2018;
13(8):e0202087. Published 2018 Aug 23. doi:10.1371/journal.pone.0202087
"""

#MIT License
#
#Copyright (c) Gordon Stevenson 2018-2020
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import numpy as np
import scipy.io as sio
import scipy.stats as sstats
import os
from icc import icc
from shape_icc import shape_icc

if __name__ == '__main__':
    #test and compare
    
    if os.path.isfile('data.mat') == False:
        raise FileNotFoundError

    data = sio.loadmat('./data.mat')
    data = data['data']

    Ny,Nx,Npts,Nraters,Nreps = data.shape

    area_icc_arr = np.zeros([Nreps,1])
    shape_icc_arr = np.zeros([Nreps,1])

    # calculate shape and area ICC for each repetition of the simulated experiment
    for rr in np.arange(0,Nreps):    
        print('Analyzing rep {} of {}\n'.format(rr,Nreps))    
        
        measured_shapes = data[:,:,:,:,rr]
        measured_areas = np.squeeze(np.sum(np.sum(measured_shapes,0),0))
        
        area_icc_arr[rr] = icc(measured_areas)[0]
        shape_icc_arr[rr] = shape_icc(measured_shapes)[0] # shape ICC for all shapes for this rep
    print('area icc {} \t shape icc {}'.format(np.mean(area_icc_arr),np.mean(shape_icc_arr)))
