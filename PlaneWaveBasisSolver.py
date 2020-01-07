"""Solve shrodinger equation in plane-wave basis"""
##############################################################################################################
##############################################################################################################
import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.integrate as integrate
period=50
V0=8
##############################################################################################################
##############################################################################################################
N=600 #number of points
a=np.pi # periodic interval
##############################################################################################################
#SECTION 1
##############################################################################################################
def kvalues(a,N):
    kvalues=np.zeros((2*N))
    for i in range(-N,N):
        k=((np.pi)/a)*(i)
        kvalues[i]=k
    return kvalues 
###############################################################################################################
def potentialfunction(x):
    return V0*np.cos(period*x)
###############################################################################################################
def xmesh():
    xmesh=np.zeros((2*N))
    for m in range(2*N):
        x=((a/N)*m)-(a)
        xmesh[m]=x
    return xmesh
###############################################################################################################
def potential(xmesh):
    potential=np.zeros((2*N))
    for m in range(2*N):
        x=((a/N)*m)-(a)
        potential[m]=potentialfunction(x)
    #print(potential)
    return potential
###############################################################################################################
#SECTION 2
###############################################################################################################
def DFT(h):
    '''
    
    INPUTS
    h: mesh of potential points
    
    OUTPUTS
    '''
    dft = np.zeros(len(h), dtype=complex)
    for i in range(len(h)):
        for j in range(len(h)):
            dft[i] += h[j] * np.exp(2*np.pi*1j*i*j/len(h))
    return dft
###############################################################################################################
def FT(potential):
    FTpotential=np.real(DFT(potential))
    return FTpotential

###############################################################################################################
def T(N,kvalues):
    h=1 #hbar
    m=1
    T=np.zeros((2*N,2*N))
    for j in range((2*N)-1):
        for i in range((2*N)-1):
            if j == i:                                                   
                T[i,j]=((h**2)/(2*m))*(kvalues[j])**2
    return T
###############################################################################################################
def V(N,FTpotential,kvecotors):
    V=np.zeros((2*N,2*N),dtype=complex)
    for j in range(2*N-1):
       for i in range(2*N-1):
           V[i,j]=(1/np.sqrt(a))*np.real(FTpotential[i-j])
    return V
#################################################################################################################                                                      
def H(T,V):
    """This function generates the Hamoltonian matrix for the shrodigner equation that is
        is to be diagnaolized and haveit egienvector calulated
    INPUTS:
    OUTPUTS:
    """
    H=T+V
    return H
#################################################################################################################
#SECTION 3
#################################################################################################################
def calculate_eigenvalues(H):
    """This fucntion calculates the eigenvectors and eigenvalues of the Hamilotonian matrix
    INPUTS: H (NxN hamiltonian matrix)
    OUTPUTS: eigenvalues, eigenvecotrs (numy array of dimesnion N)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors
#################################################################################################################
def revert_eigenvectors(eigenvectors,N):
    reverted_eigenvectors=np.zeros((2*N,2*N))
    for i in range(2*N):
        reverted_eigenvectors[:,i]=np.real(np.fft.ifft(eigenvectors[:,i]))
    return reverted_eigenvectors
##############################################################################################################
def normalize_eigenvectors(eigenvalues,eigenvectors,xmesh,N,interval):
    """This fucntion normalizes the eigenvectors and also
       produces a probability density fucntion(PDF)
    INPUTS:eigenvectors (eigenvector wavefuncitons)
           xmesh (array of x values)
           N (number of steps)
           interval(interval of evaluation)
    OUTPUTS: pdf (probability density function)
             norm_eigenvactors (normalized eigenvectors)
    """
    (a,b) = interval
    
    pdf=np.zeros((2*N,2*N))
    norm_eigenvectors=np.zeros((2*N,2*N))
    norm_eigenvalues=np.zeros((1,2*N))
    for i in range(2*N):
        pdf[:,i] = np.real(np.multiply(eigenvectors[:,i],eigenvectors[:,i]))
        normconstant = 1/(integrate.trapz(pdf[:,i],xmesh))**0.5
        pdf[:,i] = (normconstant**2)*pdf[:,i]
        norm_eigenvectors[:,i]= normconstant*eigenvectors[:,i]
        norm_eigenvalues[0,i] = eigenvalues[i]


    return pdf,norm_eigenvectors,norm_eigenvalues
##############################################################################################################
def calculate_expectation_values(eigenvectors,xmesh,N):
    """This fucntion calculates the expectations values for the unitary, space, and momentum operators
       and stores each expectation value for each egienvecotr in an array called the expectation value
       matrix
    INPUTS:eigenvectors: egienvectors in form of numpy array outputed by normalize_eigenvectors function
           xmesh: xmesh grid points in form of numpy array
           N: number of steps
    OUTPUTS:expectation_value_matrix in form of a numpy matrix
    """
    identity=np.ones((1,len(eigenvectors[:,0]))) # Identity matrix for testing

    expectation_value_matrix=np.zeros((2*N-2,3))   
    for i in range(2*N-2):
        expectation_value_matrix[i,0] = integrate.trapz((eigenvectors[:,1]*eigenvectors[:,1]),xmesh)
        expectation_value_matrix[i,1] = integrate.trapz((eigenvectors[:,1]*xmesh*eigenvectors[:,1]),xmesh)
        expectation_value_matrix[i,2] = integrate.trapz((eigenvectors[:,1]*np.gradient(eigenvectors[:,1])),xmesh)
    return expectation_value_matrix
##############################################################################################################
def plot_wavefunctions(xmesh,interval,N,eigenvectors,pdf): 
    """This fucntion plots the wavefunctions/eigenvectors
       and the probabiltiy density funciton(PDF)
    INPUTS:xmesh: xmesh grid points in form of numpy array
           interval:interval of evaluation
           N:number of steps
           V:specified potential
           eigenvectors:eigenvector in form of numpy array outputted by normalize_eigenvectors functionr
           pdf: probability density function in form of numpy matrix outputed by normalize-eigenvectors fucniton
    OUTPUTS: returns None
    """ 
    for i in range(10):
        psi=[]
        for val in eigenvectors[:,i]:
            psi.append(val)
    


        plt.plot(xmesh,psi)
        plt.xlabel('x', fontsize=20, color='black')
        plt.ylabel('psi', fontsize=20, color='black')
        plt.show()
      
    return None
##############################################################################################################
def graph_eigenvalues(N, eigenvalues,interval):
    """This function produces a graph of the anayltical and numerical energy eiganvalues
    INPUTS:N, number of steps
           eigenvalues: numerically calculated eigenvalues
           interval: interval of evaluation
    OUTPUTS:none
    """
    n=[]
    E_numeric=[]
    for i in range(N):
        n.append(i)
        E_numeric.append(eigenvalues[i])
    plt.plot(n,E_numeric,'b')
    plt.xlabel('n', fontsize=20, color='black')
    plt.ylabel('E', fontsize=20, color='black')
    plt.show()
    return None
##############################################################################################################
def run(a,N):
    #RUN_SECTION1---------------------------------------------------------------------------------------------
    kvecs=kvalues(a,N)
    potential1=potential(xmesh())
    #RUN_SECTION2---------------------------------------------------------------------------------------------
    FTpotential=FT(potential1)
    Tmatrix=T(N,kvecs)
    Vmatrix=V(N,FTpotential,kvecs)
    Hmatrix=H(Tmatrix,Vmatrix)
    print(Hmatrix)
    #---------------------------------------------------------------------------------------------------------
##    FTpotential_real = np.real(FTpotential)
##    plt.plot(kvecs,FTpotential_real)
##    plt.show()
##    plt.plot(xmesh(),potential1)
##    plt.show()
    #RUN_SECTION3---------------------------------------------------------------------------------------------
    eigenvalues, eigenvectors = calculate_eigenvalues(Hmatrix)
    reverted_eigenvectors = revert_eigenvectors(eigenvectors,N)
    pdf,norm_eigenvectors,norm_eigenvalues = normalize_eigenvectors(eigenvalues,reverted_eigenvectors,xmesh(),N,(-a,a))
    plot_wavefunctions(kvecs,(-a,a),N,norm_eigenvectors,pdf)
    graph_eigenvalues(N, eigenvalues,(-a,a))
    return None
###################################################################################################################

run(a,N)
                       



