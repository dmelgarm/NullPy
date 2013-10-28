'''
Diego Melgar 10/2013, dmelgarm@ucsd.edu

Analysis and modification of slip ivnersions through the null-space shuttles
method. This module reads in slip inversion results from the rtokada runs. The 
basic method is a 2 step process. First figure out how much of the joint inversion
you can place in the generalized nullspace of the coseismic offsets. This gives
you a "conservative" model. From that conservative model analyze, in the second 
step how much you can palce in the nullspace of teh wave gauges. This will
maximize the fit to both datasets.

Comments:
Rememeber young squire, the observed and synthetic data saved to the files are 
all in SI units. However, the coseismic GFs are in mm (yuck) and any forward
computation with it will produce offsets in mm.

References:
    
    Deal,M.N. and G. Nolet, Nullspace Shuttles, Geophys. J. Int, 124, 372-380
    
'''


##############    GLOBALS: Paths and names of things        ###################
datapath='/Users/dmelgarm/Documents/Research/NullPy/data/'
coseisGF='green_small_kal.mat'
waveGF='tohoku_wvGF_50min.mat'
coseisfile='RTOkada_kal_opt.0001.disp'
jointcoseisfile='RTOkada_kalwv50min_opt.0001.disp'
coseismodelfile='RTOkada_kal_opt.0001.slip'
jointmodelfile='RTOkada_kalwv50min_opt.0001.slip'
wavesfile='RTOkada_kalwv50min_opt.0001.wave'
smoothfile='reg.mat'
coseis_lambda=0.01
joint_lambda=0.145
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
    #set tsunami strike sip GFs to zero
    i=np.arange(Gwv.shape[1]/2)*2
    #Gwv[:,i]=0
    #Regularization matrix
    iobuff=loadmat(datapath+smoothfile)
    T=iobuff['T']    
    #Read coseismic data and synthetics
    coseis=np.loadtxt(datapath+coseisfile)
    #Assign to
    stalat=coseis[:,2]
    stalon=coseis[:,3]
    n=coseis[:,4]
    e=coseis[:,5]
    u=coseis[:,6]
    ns=coseis[:,7]
    es=coseis[:,8]
    us=coseis[:,9]
    numsta=len(n)
    d=neu2d(n,e,u)
    ds=neu2d(ns,es,us)
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
    #Read coseismic data and synthetics
    coseis=np.loadtxt(datapath+jointcoseisfile)
    n=coseis[:,4]
    e=coseis[:,5]
    u=coseis[:,6]
    ns=coseis[:,7]
    es=coseis[:,8]
    us=coseis[:,9]
    dwv=neu2d(n,e,u)
    dwvs=neu2d(ns,es,us)
    joint_mod=np.loadtxt(datapath+jointmodelfile)
    #Assign to variables
    mss=joint_mod[:,0] #Strike-Slip
    mds=joint_mod[:,1] #Dip-Slip
    #Combine into single vector
    mfilt=np.concatenate((mss,mds))
    #And interleave them
    i=np.arange(mss.size)
    iss=(i*2) #SS are even elements
    ids=(i*2)+1 #DS are 
    mfilt[iss]=mss
    mfilt[ids]=mds
    #Load tsunami wave gauges
    waves=np.loadtxt(datapath+wavesfile)
    wobs=waves[:,1]
    wsyn=waves[:,2]
    #Make output vectors

    #And done...
    return stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,T
    
    
#--------------------          Shuttle operator       --------------------------
def shuttle(m,mfilt,G,neigen):
    '''
    Given a model m and a 'filtered' model mfilt, construct from the Green 
    functions matrix, G, of m the null space shuttle operator using the first ns
    eigen-values. Apply the operator and return the model mconservative
    
    Usage:
        mconservative=shuttle(m,mfilt,G,neigen)
        
    The goal is to find the eigen vectors of G'*G. This is done by noting that
    the columns of the right SINGULAR vectors of G are the eigen-vectors of G'G.
    '''
    
    from scipy.linalg import svd
    from numpy import eye,dot
    
    #Compute SVD
    U,s,V=svd(G,full_matrices=False)
    #Keep only neigen columns
    #V=V.transpose()
    Vk=V[:,0:neigen]
    #delta
    dm=mfilt-m
    #shuttle
    SH=eye(Vk.shape[0])-dot(Vk,Vk.transpose())
    #Make model
    mconservative=m+dot(SH,dm)
    return mconservative
    
    
    
    
    
    
def svdanalysis(misfit=False,plots=False):
    '''
    Get the svd and make some plots of its behavior
    '''
    from scipy.linalg import svd
    import numpy as np
    from matplotlib import pyplot as plt
    #Get all the stuff
    stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,T = load_data()
    GS=np.concatenate((G,coseis_lambda*T))
    #Compute the svd
    U,s,V=svd(GS,full_matrices=False)
    nsvd=np.zeros(s.shape)
    #Compute misift changes?
    if misfit:
        #Compute misfit as we get rid of singular values
        dobs=d
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
    
def recshut(d,wobs,m,mfilt,G,Gwv,neigen,neigen2,plots=True,stalon=1,stalat=1):
    from numpy import dot,arange,arctan,rad2deg
    import matplotlib.pyplot as plt
    mconservative=shuttle(m,mfilt,G,neigen)
    mconservative2=shuttle(mfilt,mconservative,Gwv,neigen2)
    #Compute conservative coseismics
    dc=dot(G,mconservative)/1000
    vr=vred(d,dc)
    dc2=dot(G,mconservative2)/1000
    vr2=vred(d,dc2)
    print ' --> VR='+str(vr2)+'%'
    vr=vr2
    nc,ec,uc=d2neu(dc2)
    #But what about the tsunami bro?
    wconserv=dot(Gwv,mconservative)/1000
    wconserv2=dot(Gwv,mconservative2)/1000
    newrms=rms(wobs,wconserv)
    newrms2=rms(wobs,wconserv2)
    r=newrms2
    print ' --> RMS='+str(r)+'m' 
    #Original data
    n,e,u=d2neu(d)
    if plots==True:
        #Plot em
        plt.close('all')
        plt.figure()
        plt.quiver(stalon,stalat,e,n,width=0.0025)
        plt.quiver(stalon,stalat,ec,nc,width=0.0025,color='blue')
        plt.axis('equal')
        plt.ylim((32,48))
        plt.xlim((132,150))
        plt.grid()
        plt.figure()
        iss=arange(m.size/2)*2
        ids=arange(m.size/2)*2-1
        plt.subplot(211)
        plt.plot(m[iss]/1000)
        plt.plot(mfilt[iss]/1000)
        plt.plot(mconservative[iss]/1000)
        plt.plot(mconservative2[iss]/1000)
        plt.title('Strike Slip component')
        plt.legend(['coseismic','joint','conservative 1','conservative 2'])
        plt.ylim((-10,80))
        plt.grid()
        plt.subplot(212)
        plt.plot(m[ids]/1000)
        plt.plot(mfilt[ids]/1000)
        plt.plot(mconservative[ids]/1000)
        plt.plot(mconservative2[ids]/1000)
        plt.title('Dip Slip component')
        plt.legend(['coseismic','joint','conservative 1','conservative 2'])
        plt.grid()
        plt.ylim((-10,80))
        plt.figure()
        plt.scatter(arange(len(m[ids])),rad2deg(arctan(m[ids]/m[iss])))
        plt.scatter(arange(len(m[ids])),rad2deg(arctan(mconservative[ids]/mconservative[iss])),color='red')
    return vr,r
    
def runall():
    from numpy import zeros
    stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,T = load_data()
    neigs=G.shape[1]
    vr=zeros(G.shape[1]**2)
    r=zeros(G.shape[1]**2)
    coseiseigs=zeros(G.shape[1]**2)
    wveigs=zeros(G.shape[1]**2)
    n=0
    for k1 in range(neigs):
        for k2 in range(neigs):
            print str(n)+'/'+str(neigs**2)
            vr[n],r[n]=recshut(d,wobs,m,mfilt,G,Gwv,neigs-k1,neigs-k2,plots=False)
            coseiseigs[n]=neigs-k1
            wveigs[n]=neigs-k2
            n=n+1
    return vr,r,coseiseigs,wveigs
    
        
    
    
    
    
#--------------------------      Random Tools      -----------------------------
def neu2d(n,e,u):
    '''
    Convert n,e,u numpy vectors into a single vector ordered like e1,n1,u1,e2,
    n2,u2 and so on.
    
    Usage:
        d=neu2d(n,e,u)
    '''
    from numpy import arange,zeros
    i=arange(n.size)
    d=zeros(n.size*3,)
    d[i*3]=e
    d[(i*3)+1]=n
    d[(i*3)+2]=u
    return d
    
def d2neu(d):
    from numpy import arange
    i=arange(len(d)/3)
    n=d[(i*3)+1]
    e=d[(i*3)]
    u=d[(i*3)+2]
    return n,e,u
    
def vred(dobs,dsyn):
    delta=(dobs-dsyn)**2
    vr=(1-(sum(delta)/sum(dobs**2)))*100
    return vr
    
def rms(dobs,dsyn):
    from numpy import sqrt
    delta=(dobs-dsyn)**2
    r=sqrt(sum(delta)/len(delta))
    return r
