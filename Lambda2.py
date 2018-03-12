from __future__ import division
import numpy as np

def Lambda2(qu, qv, qw, dx, dy, dz):

    #% Inputs:
    #%   qu,qv,qw are three components of the velocity field in block data
    #%   format 
    #%   qx,qy,qz are cartesian coordinates corresponding to the velcity data
    #%   
    #% Outputs:
    #%   Negative lambda2 belows to a vortex by definition
    
    DUDX, DUDY, DUDZ = np.gradient(qu, dx, dy, dz)
    DVDX, DVDY, DVDZ = np.gradient(qv, dx, dy, dz)
    DWDX, DWDY, DWDZ = np.gradient(qw, dx, dy, dz)
    ST00=1/2*(DUDX+DUDX)
    ST01=1/2*(DUDY+DVDX)
    ST02=1/2*(DUDZ+DWDX)
    ST10=1/2*(DVDX+DUDY)
    ST11=1/2*(DVDY+DVDY)
    ST12=1/2*(DVDZ+DWDY)
    ST20=1/2*(DWDX+DUDZ)
    ST21=1/2*(DWDY+DVDZ)
    ST22=1/2*(DWDZ+DWDZ)
    VT00=1/2*(DUDX-DUDX)
    VT01=1/2*(DUDY-DVDX)
    VT02=1/2*(DUDZ-DWDX)
    VT10=1/2*(DVDX-DUDY)
    VT11=1/2*(DVDY-DVDY)
    VT12=1/2*(DVDZ-DWDY)
    VT20=1/2*(DWDX-DUDZ)
    VT21=1/2*(DWDY-DVDZ)
    VT22=1/2*(DWDZ-DWDZ)
    STS00=ST00*ST00+ST01*ST10+ST02*ST20+VT00*VT00+VT01*VT10+VT02*VT20
    STS01=ST00*ST01+ST01*ST11+ST02*ST21+VT00*VT01+VT01*VT11+VT02*VT21
    STS02=ST00*ST02+ST01*ST12+ST02*ST22+VT00*VT02+VT01*VT12+VT02*VT22
    STS10=ST10*ST00+ST11*ST10+ST12*ST20+VT10*VT00+VT11*VT10+VT12*VT20
    STS11=ST10*ST01+ST11*ST11+ST12*ST21+VT10*VT01+VT11*VT11+VT12*VT21
    STS12=ST10*ST02+ST11*ST12+ST12*ST22+VT10*VT02+VT11*VT12+VT12*VT22
    STS20=ST20*ST00+ST21*ST10+ST22*ST20+VT20*VT00+VT21*VT10+VT22*VT20
    STS21=ST20*ST01+ST21*ST11+ST22*ST21+VT20*VT01+VT21*VT11+VT22*VT21
    STS22=ST20*ST02+ST21*ST12+ST22*ST22+VT20*VT02+VT21*VT12+VT22*VT22
    A=-(STS00 + STS11 + STS22)
    B=-(STS01*STS10 + STS02*STS20 + STS12*STS21 - STS00*STS11 - STS11*STS22 - STS00*STS22)
    C=-(STS00*STS11*STS22 + STS01*STS12*STS20 + STS02*STS21*STS10 - STS00*STS12*STS21 - STS11*STS02*STS20 - STS22*STS01*STS10)
    lambda2=2*np.cos(np.arccos((9*A*B - 27*C - 2*A**3)/54/np.sqrt(-((3*B - A**2)/9)**3))/3 - 2*np.pi()/3)*np.sqrt(-(3*B - A**2)/9) - A/3
    return lambda2