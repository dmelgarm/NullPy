'''
Diego Melgar 10/2013, dmelgarm@ucsd.edu

Analysis and modification of slip ivnersions through the null-space shuttles
method

Comments:
Rememeber young squire, the observed and synthetic data saved to the files are 
all in SI units. However, the coseismic GFs are in mm (yuck) and any forward
computation with it will produce offsets in mm.
'''


##############    GLOBALS: Paths and names of things        ###################
datapath='/Users/dmelgarm/Documents/Research/NullPy/data/'
coseisGF='green_small_kal.mat'
waveGF='tohoku_wvGF_50min.mat'
coseisfile='RTOkada_kal_opt.0001.disp'
coseismodelfile='RTOkada_kal_opt.0001.slip'
################################################################################



#--------------------        Load the data     ---------------------------------
def load_data():
    '''
    Load the data from the RTOkada inversions into numpy arrays
    
    Usage:
        stalat,stalon,n,e,u,ns,es,us,G,Gwv = load_data()
    '''
    
    from scipy.io import loadmat
    import numpy as np
    
    #Read matlab files for Green's Functions
    iobuff=loadmat(datapath+coseisGF)
    G=iobuff['G']
    iobuff=loadmat(datapath+waveGF)
    Gwv=iobuff['Gwv']
    #Read coseismic data and synthetics
    coseis=np.loadtxt(datapath+coseisfile)
    #Assign to variables
    stalat=coseis[:,2]
    stalon=coseis[:,3]
    n=coseis[:,4]
    e=coseis[:,5]
    u=coseis[:,6]
    ns=coseis[:,7]
    es=coseis[:,8]
    us=coseis[:,9]
    numsta=len(n)
    #Read coseismic model
    coseis_mod=np.loadtxt(datapath+coseismodelfile)
    #Assign to variables
    mss=coseis_mod[:,0] #Strike-Slip
    mds=coseis_mod[:,1] #Dip-Slip
    #Combine into single vector
    m=np.concatenate((mss,mds))
    #And interleave them
    i=np.arange(mss.size)
    iss=(i*2) #SS are even elements
    ids=(i*2)+1 #DS are 
    m[iss]=mss
    m[ids]=mds
    #Now check you are not dumb, make the forward computation and compare to synthetics
    dtest=np.dot(G,m)   #vector is e1,n2,u1,e2,n2,u2,...
    i=np.arange(numsta)
    ntest=dtest[(i*3)+1]
    etest=dtest[(i*3)]
    utest=dtest[(i*3)+2]
    dn=abs(ntest-ns*1000)
    de=abs(etest-es*1000)
    du=abs(utest-us*1000)
    if dn.max()>1 or de.max()>1 or du.max()>1: #You screwed something up
        print '--> Forward computed data does NOT match synthetic data on file!'
    else:
        print '--> Sanity check OK, forward data matches data on file'
    #Load Joint Inversion as well
    Gwv=1
    #Load tsunami only inversion?
    pass
    return stalat,stalon,n,e,u,ns,es,us,m,G,Gwv
    
def svdanalysis(misfit=False,plots=False):
    '''
    Get the svd and make some plots of its behavior
    '''
    from scipy.linalg import svd
    import numpy as np
    from matplotlib import pyplot as plt
    #Get all the stuff
    stalat,stalon,n,e,u,ns,es,us,m,G,Gwv = load_data()
    #Compute the svd
    U,s,V=svd(G,full_matrices=False)
    nsvd=np.zeros(s.shape)
    #Compute misift changes?
    if misfit:
        #Compute misfit as we get rid of singular values
        dobs=neu2d(n,e,u)
        VR=np.zeros(s.shape)
        s0=s.copy()
        for k in range(len(s)-1):
            k=k+1
            nsvd[k-1]=len(s)-k  #No of values used
            s0[0-k]=0 #Set to zero
            S0=np.diag(s0)    #Make into matrix
            GS=np.dot(U,np.dot(S0,V)) #Reconstruct
            dsvd=np.dot(GS,m)/1000 #Convert to m
            delta=(dobs-dsvd)**2
            VR[k-1]=(1-(sum(delta)/sum(dobs**2)))*100
            #print 'nsvd='+str(nsvd[k-1])+','+'VR='+str(VR[k-1])+'%'
        if plots:
            plt.close("all")
            plt.figure()
            plt.subplot(211)
            plt.plot(s)
            plt.xlabel('Singular value No.')
            plt.ylabel('Singular value')
            plt.grid()
            plt.subplot(212)
            plt.plot(nsvd,VR)
            plt.ylabel('Variance Reduction(%)')
            plt.xlabel('Total number of singular values used')
            plt.grid()
    #Make plots
    

def neu2d(n,e,u):
    from numpy import arange,zeros
    i=arange(n.size)
    d=zeros(n.size*3,)
    d[i*3]=e
    d[(i*3)+1]=n
    d[(i*3)+2]=u
    return d