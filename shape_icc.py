#! /usr/bin/python

"""
Python implementation of the ShapeICC method devised by Travis Smith from
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

import scipy.io as sio
import scipy.stats as sstats
import numpy as np

def shape_icc(M):
    """SHAPEICC Shape intraclass correlation coefficient.
    Calculates ICC is for a two-way, fully crossed random effects model.
    This type of ICC is appropriate to describe the absolute agreement 
    among shape measurements from a group of k raters, randomly selected 
    from the population of all raters, made on a set of n items.  
    Shrout and Fleiss: ICC(2,1)
    McGraw and Wong:   ICC(A,1)

    M is the stack of N-dimensional rasterized shape measurements
        The dimensions of M are Ny x Nx x ... x n x k, where
            Ny x Nx x ... is the size of each N-dimensional rasterized shape
            n is the # of subjects / groups
            k is the # of raters

        For example, with 2-D shapes,
        M(:,:,2,3) would be the shape corresponding to subject 2, rater 3
    """
    # setup size
    sizeM = M.shape
    #k - num raters; n - num subjects
    k = sizeM[-1]
    n = sizeM[-2]
    Npix = np.prod(sizeM[0:-2])  #Npix = Ny * Nx * ... = total # of pixels in each shape

    M = np.reshape(M,[Npix, n, k])  #vectorize each shape into an Npix-length column array
    M = np.single(M)  #in case M is logical, need it to be single to do floating-point ops on it

    # compute means
    u1 = np.squeeze(np.mean(M, axis =1))  # Npix x k
    u2 = np.mean(M, axis =2)  # Npix x n
    u = np.mean(u1, axis =1)  # Npix x 1

    SS = 0
    for ii in np.arange(n):
        for jj in np.arange(k):
            d = np.abs(M[:,ii,jj] - u)
            SS = SS + (np.sum(d) ** 2)

    MSR = 0
    for ii in np.arange(n):
        d = abs(u2[:,ii] - u)
        MSR = MSR + sum(d[:]) ** 2
    MSR = k/(n-1) * MSR

    MSC = 0
    for jj in np.arange(k):    
        d = abs(u1[:,jj] - u)
        MSC = MSC + (sum(d[:]) ** 2)
    MSC = n/(k-1) * MSC

    MSE = (SS - (n-1)*MSR - (k-1)*MSC) / ((n-1)*(k-1))
    icc = (MSR - MSE) / (MSR + (k-1)*MSE + k/n*(MSC-MSE))
    
   #from ICC(A,1) - in IRR package
    ns = n
    nr = k
    coeff = (MSR - MSE)/(MSR + (nr - 1) * MSE + 
                  (nr/ns) * (MSC - MSE))
    r0 = 0
    a = (nr * r0)/(ns * (1 - r0))
    b = (nr * r0 * (ns - 1))/(ns * (1 - r0))    
    Fvalue = MSR/(a * MSC + b * MSE)

    alpha = 1 - 0.95
    a = (nr * coeff)/(ns * (1 - coeff))
    b = 1 + (nr * coeff * (ns - 1))/(ns * (1 - coeff))
    v = (a * MSC + b * MSE)**2/((a * MSC)**2/(nr - 1) + (b * MSE)**2/((ns - 1) * (nr - 1)))
    #p.value <- pf(Fvalue, df1, df2, lower.tail = FALSE)
    #FL <- qf(1 - alpha/2, ns - 1, v)
    #FU <- qf(1 - alpha/2, v, ns - 1)

    FL = sstats.f.ppf(1 - alpha/2, ns - 1, v)
    FU = sstats.f.ppf(1 - alpha/2, v, ns - 1)

    lbound = (ns * (MSR - FL * MSE))/(FL * (nr * MSC + (nr * ns - nr - ns) * MSE) + ns * MSR)
    ubound = (ns * (FU * MSR - MSE))/(nr * MSC + (nr * ns - nr - ns) * MSE + ns * FU * MSR)

    return icc, Fvalue, lbound, ubound
