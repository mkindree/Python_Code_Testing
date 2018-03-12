from __future__ import division
import numpy as np

def spectrumFFT_v2(U, Fs):
    # Update of spectrumFFT which was written by Jason(?): 
    # added comments
    
    N = len(U)

    # Truncate to an even number of samples
    NFFT = 2*int(np.floor(N/2))
    U = U[0:NFFT]
    
    # Decompose
    Um = np.mean(U)
    uf = U - Um
    
    
    
    # Frequency and time
    df = Fs/NFFT
    f = df*np.arange(0, NFFT/2, 1).T
#    f = np.arange(0, Fs/2+df, df).T
    omega = 2*np.pi*f
    T = (NFFT - 1)/Fs
    dt = 1/Fs
    t = np.arange(0, T, dt).T
    
    # Discrete Fourier Transform
#    DFT = 2*np.fft.fft(uf)/NFFT # Divide by N/2 to get the proper amplitudes
    # Symmetric
#    DFT = DFT[0:NFFT/2+1]
    
    # Power Spectral Density
    PSD = np.conj(np.fft.fft(uf))*np.fft.fft(uf)/(2*np.pi*NFFT*Fs) #2pi because of angular frequency
#    PSD = np.abs(np.fft.fft(uf))^2/(2*np.pi*NFFT*Fs) 
    # Symmetric
    PSD = PSD[0:int(NFFT/2)]
    # We are only considering half of the symmetric spectrum so we need to 
    # double the energy, 0 frequency and Nyquist frequency do not occur twice
    PSD[1:] = 2*PSD[1:]
    PSD = np.real(PSD)

    return f, PSD
