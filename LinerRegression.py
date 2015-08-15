#!/bin/python

from math import *
from numpy.random import normal,rand
import numpy as np
import matplotlib.pyplot as plt
import random

#define the Max number of ratio
MAXRATIO=10

def GetRatio(x,y,m):
    #  x -> Horizontal coordinates
    #  y -> Longitudinal coordinates
    #  m -> numbers of ratio
    #  Aa=B,use numpy.linalg.solve(A,B) get a
    n = len(x)
    B = y
    temp = [n]
    B = []

    #caculate sum of xi^j where j is [0,1,...,2m]
    for i in range( 1 , 2*m + 1 ):
        sum = 0.0
        for j in range( n ):
            sum = sum + x[j]**i
        temp.append(sum)

    #caculate sum of xi^j*yi where j is [0,1,...,m]
    for i in range( m+1 ):
        sum=0.0
        for j in range( n ):
            sum = sum + (x[j]**i)*y[j]
        B.append(sum)

    #generate matrix A
    A=[[] for i in range( m + 1 )]
    index=0
    for i in range(m + 1):
        for j in range(m + 1):
            A[i].append(temp[j + index])
        index = index + 1
    
    #solve the formule
    A=np.array(A)
    B=np.array(B)
    w=np.linalg.solve(A,B)
    return w

def ReFunction(x,w):
    #fm(x,w)=w0+w1*x^1+w2*x^2+...+wm*x^M
    n=len(w)
    sum=0.0
    for i in range(n):
        sum = sum + w[i]*x**i
    return sum
    
    
def LossFunction(x,y,w,lam):
    #loss function is 1/N*(sum((f(xi;w)-yi)^2)+lam/2*||w||^2
    #w -> ratios vector
    #lam -> lambda : ratio of loss
    n=len(x)
    L=0.0
    for i in range(n):
        L = L + (ReFunction(x[i],w) - y[i])**2
    L=L/n + (lam/2)*((len(w))**2)
    return L
    
def LinerRe(x,y,lam=0.1):
    #  x -> Horizontal coordinates
    #  y -> Longitudinal coordinates
    # lam -> lam is a small digit
    lam=float(lam)
    w_final = GetRatio( x , y , 0 )
    L_min = LossFunction( x , y , w_final , lam ) 
    for i in range(1,MAXRATIO):
        w = GetRatio(x,y,i)
        L = LossFunction( x , y , w , lam )
        if L < L_min:
            w_final = w
            L_min = L
        else:
            return w_final
    return w_final

def Plot(x,y,lam):
    #draw origin (x,y)
    plt.plot( x , y ,color='m',linestyle='',marker='.')
    #draw new (x,y)
    x=[float(i) for i in x]
    y=[float(i) for i in y]
    #get ratio
    w=LinerRe( x , y , lam )

    y_new=[]
    x.sort()
    x_min=x[0]
    x_max=x[len(x)-1]
    x_new=np.arange(x_min,x_max,(x_max - x_min)/(10*len(x)))
    for i in x_new:
        y_new.append(ReFunction( i , w ))
    plt.plot( x_new , y_new ,color='g' , linestyle='-' , marker = '')
    plt.show()
 
def main():
    #change the x ande y,and this will draw the pic autoly by scale
    #for example :by JairusChan on csdn


    ##############################################
    xa=np.arange(-1,1,0.02)
    ya=[((a*a-1)**3+0.5)*np.sin(a*2) for a in xa]
    i=0
    x=[]
    y=[]
    for xx in xa:
        yy=ya[i]
        d=float(random.randint(60,140))/100
        i = i + 1
        x.append(xx*d)
        y.append(yy*d)
    ##############################################


    Plot( x , y , 0.0)



if __name__ == '__main__':
    main()
    
