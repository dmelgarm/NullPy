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
datapath='/Users/dmelgarm/Research/Data/Tohoku/RTOkada/Results/opt/'
gfpath='/Users/dmelgarm/Research/Data/Tohoku/RTOkada/'
coseisGF='green_small.mat'
waveGF='tohoku_wvGFnoSS_50min.mat'
coseisfile='RTOkada_allgps_opt.0001.disp'
jointcoseisfile='RTOkada_gpswv_opt.0001.disp'
coseismodelfile='RTOkada_allgps_opt.0001.slip'
jointmodelfile='RTOkada_gpswv_opt.0001.slip'
tsunamimodelfile='RTOkada_wvonly_opt.0001.slip'
wavesfile='RTOkada_gpswv_opt.0001.wave'
smoothfile='reg.mat'
coseisweights='Wallgps.mat'
waveweights='Wwv.mat'
outputdir='/Users/dmelgarm/Research/NullPy/output/'
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
    iobuff=loadmat(gfpath+coseisGF)
    G=iobuff['G']
    iobuff=loadmat(gfpath+waveGF)
    Gwv=iobuff['Gwv']  
    #Read data weights matrix
    iobuff=loadmat(gfpath+coseisweights)
    Wgps=iobuff['Wgps']
    Wgps=np.array(Wgps,dtype='>d') 
    iobuff=loadmat(gfpath+waveweights)
    Wwv=iobuff['Wwv']
    Wwv=np.array(Wwv,dtype='>d') 
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
    #Load Tsunami Inversion as well
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
    joint_mod=np.loadtxt(datapath+tsunamimodelfile)
    #joint_mod=np.loadtxt(datapath+jointmodelfile)
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
    return stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,Wgps,Wwv
    
    
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
    from numpy import eye,dot,array
    
    #Compute SVD
    G=array(G,dtype='>d')  #Some bug in numpy or scipy misreads the endianess
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
    stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,Wgps,Wwv = load_data()
    #Compute the svd
    G=np.array(G,dtype='>d')  #Some bug in numpy or scipy misreads the endianess
    U,s,V=svd(G,full_matrices=False)
    nsvd=np.zeros(s.shape)
    #Compute misift changes?
    if misfit:
        #Compute misfit as we get rid of singular values
        dobs=d
        VR=np.zeros(s.shape)
        VRshut=np.zeros(s.shape)
        s0=s.copy()
        for k in range(len(s)-1):
            print k
            nsvd[k-1]=k  #No of values used
            mconservative=shuttle(m,mfilt,G,len(s)-k)
            dshut=np.dot(G,mconservative)/1000
            delta=(dobs-dshut)**2
            VRshut[k-1]=(1-(sum(delta)/sum(dobs**2)))*100
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
            plt.figure()
            plt.plot(nsvd,VRshut)
            plt.ylabel('Variance Reduction(%)')
            plt.xlabel('Size of generalized nullspace')
            plt.grid()
    #Make plots
    
def recshut(d,wobs,m,mfilt,G,Wgps,Gwv,Wwv,neigen,neigen2=0,plots=True,stalon=1,stalat=1,numpass=2):
    '''
    
    '''
    from numpy import dot,arange,arctan,rad2deg
    from numpy.linalg import norm
    import matplotlib.pyplot as plt
    mconservative=shuttle(m,mfilt,G,neigen)
    if numpass==2:
        mconservative2=shuttle(mfilt,mconservative,Gwv,neigen2)
        misfitcoseis2=norm(dot(Wgps,dot(G,mconservative2)/1000)-dot(Wgps,d))
        misfitwaves2=norm(dot(Wwv,dot(Gwv,mconservative2)/1000)-dot(Wwv,wobs))
    else:
        misfitcoseis2=0
        misfitwaves2=0
        mconservative2=mconservative
    #Compute conservative coseismics
    misfitcoseis1=norm(dot(Wgps,dot(G,mconservative)/1000)-dot(Wgps,d))
    misfitwaves1=norm(dot(Wwv,dot(Gwv,mconservative)/1000)-dot(Wwv,wobs))
    print ' --> 1st pass L2coseis ='+str(misfitcoseis1) 
    print ' --> 1st pass L2waves ='+str(misfitwaves1)
    print ' --> 2nd pass L2coseis ='+str(misfitcoseis2) 
    print ' --> 2nd pass L2waves ='+str(misfitwaves2)
    #Original data
    n,e,u=d2neu(d)
    if plots==True:
        #Plot em
        plt.close('all')
        plt.figure()
        plt.quiver(stalon,stalat,e,n,width=0.0025)
        plt.quiver(stalon,stalat,es,ns,width=0.0025,color='blue')
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
    return mconservative2



def runone(neigs,plots=False):    
    from numpy import savetxt,loadtxt,dot
    from numpy.linalg import norm
    run='allgps_joint_opt'
    #read a previous ivnersion file for filef ormat
    outmodel=loadtxt(datapath+coseismodelfile)
    stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,Wgps,Wwv = load_data()
    mshut=recshut(d,wobs,m,mfilt,G,Wgps,Gwv,Wwv,neigen=neigs,plots=False,numpass=1)
    #Get misifit
    misfitcoseis=norm(dot(Wgps,dot(G,mshut)/1000)-dot(Wgps,d))
    misfitwaves=norm(dot(Wwv,dot(Gwv,mshut)/1000)-dot(Wwv,wobs))
    #save to file
    ss,ds=m2ssds(mshut)
    outmodel[:,0]=ss
    outmodel[:,1]=ds
    savetxt(outputdir+run+'.0001.slip',outmodel)
    #and make a log
    print neigs
#    log= '''neigs1 %(neigs)s
#neigs2 0
#L2coseis %(misfitcoseis)s
#L2waves %(misfitwaves)s
#    ''' % {'neigs1':str(neigs),'misfitcoseis':str(misfitcoseis),'misfitwaves':str(misfitwaves)}
#    f = open(outputdir+run+'.0001.log', 'w')
#    f.write(log)
#    f.close()
    plot_results(stalon,stalat,mshut,G,Gwv,d,wobs)
    
            
def runall():
    from numpy import savetxt,loadtxt,dot
    from numpy.linalg import norm
    run='allgps_joint_opt'
    #read a previous ivnersion file for filef ormat
    outmodel=loadtxt(datapath+coseismodelfile)
    stalat,stalon,d,ds,dwv,dwvs,wobs,wsyn,m,mfilt,G,Gwv,Wgps,Wwv = load_data()
    neigs=G.shape[1]
    n=0
    mult=5
    for k1 in range(0,neigs,mult):
        for k2 in range(0,neigs,mult):
            print str(n)+'/'+str((neigs/5)**2)
            neigs1=neigs-k1
            neigs2=neigs-k2
            print neigs1,neigs2
            mshut=recshut(d,wobs,m,mfilt,G,Wgps,Gwv,Wwv,neigs1,neigs2,plots=False)
            #Get misifit
            misfitcoseis=norm(dot(Wgps,dot(G,mshut)/1000)-dot(Wgps,d))
            misfitwaves=norm(dot(Wwv,dot(Gwv,mshut)/1000)-dot(Wwv,wobs))
            #save to file
            ss,ds=m2ssds(mshut)
            outmodel[:,0]=ss
            outmodel[:,1]=ds
            savetxt(outputdir+run+'.'+str(n).rjust(6,'0')+'.slip',outmodel)
            #and make a log
            log= '''neigs1 %(neigs1)s
neigs2 %(neigs2)s
L2coseis %(misfitcoseis)s
L2waves %(misfitwaves)s
        ''' % {'neigs1':str(neigs1),'neigs2':str(neigs2),'misfitcoseis':str(misfitcoseis),'misfitwaves':str(misfitwaves)}
            f = open(outputdir+run+'.'+str(n).rjust(6,'0')+'.log', 'w')
            f.write(log)
            f.close()
            n=n+1

def plot_results(lon,lat,mshut,G,Gwv,d,wobs):
    import matplotlib.pyplot as pl
    import numpy as np
    pl.close("all")
    ds=np.dot(G,mshut)/1000
    ws=np.dot(Gwv,mshut)/1000
    n,e,u=d2neu(d)
    ns,es,us=d2neu(ds)
    pl.figure()
    pl.subplot(211)
    pl.quiver(lon,lat,e,n)
    pl.quiver(lon,lat,es,ns)
    pl.axis('equal')
    pl.legend(['Obs','Syn'])
    pl.grid()
    pl.subplot(212)
    pl.quiver(lon,lat,np.zeros(n.shape),u)
    pl.quiver(lon,lat,np.zeros(n.shape)+0.1,us)
    pl.axis('equal')
    pl.legend(['Obs','Syn'])
    pl.grid()
    pl.figure()
    t=np.arange(0,wobs.shape[0]*15,15)
    pl.plot(t,wobs)
    pl.plot(t,ws)
    pl.legend(['Obs','Syn'])
    pl.grid()
    
    
                        
                                                            
def misfitplots():
    
    import glob
    import matplotlib.pyplot as pl
    import numpy as np
    from scipy.interpolate import griddata
    run='allgps_tsun'
    iolist=glob.glob(outputdir+run+'*.log')
    neigs1=np.zeros(len(iolist))
    neigs2=np.zeros(len(iolist))
    L2coseis=np.zeros(len(iolist))
    L2waves=np.zeros(len(iolist))
    for k in range(len(iolist)):
        with open(iolist[k],'r') as f:
            line=f.readlines()
            key, l = line[0].split(' ')
            neigs1[k]=int(l.rstrip())
            key, l = line[1].split(' ')
            neigs2[k]=int(l.rstrip())
            key, l = line[2].split(' ')
            L2coseis[k]=float(l.rstrip())
            key, l = line[3].split(' ')
            L2waves[k]=float(l.rstrip())
    #Now make some plots
    xi=np.arange(3,379,1)
    yi=np.arange(3,379,1)
    zi=griddata((neigs1,neigs2),L2coseis,(xi[None,:],yi[:,None]))
    pl.close("all")
    pl.figure()
    pl.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    pl.contourf(xi,yi,zi,15)
    pl.colorbar()
    pl.xlabel('Coseismic Neigs')
    pl.ylabel('Tsunami Neigs')
    pl.title('Coseismic Misfit')
    
    zi=griddata((neigs1,neigs2),L2waves,(xi[None,:],yi[:,None]))
    pl.figure()
    pl.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    pl.contourf(xi,yi,zi,15)
    pl.colorbar()
    pl.xlabel('Coseismic Neigs')
    pl.ylabel('Tsunami Neigs')
    pl.title('Tsunami Misfit')
    
    zi=griddata((neigs1,neigs2),L2waves+L2coseis,(xi[None,:],yi[:,None]))
    pl.figure()
    pl.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    pl.contourf(xi,yi,zi,15)
    pl.colorbar()
    pl.xlabel('Coseismic Neigs')
    pl.ylabel('Tsunami Neigs')
    pl.title('Total Misfit')


    
    
        
    
    
    
    
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
    
def vred(dobs,dsyn,W):
    from numpy import dot
    delta=(dot(W,dobs)-dot(W,dsyn))**2
    vr=(1-(sum(delta)/sum(dot(W,dobs)**2)))*100
    return vr
    
def rms(dobs,dsyn,W):
    from numpy import sqrt,dot
    delta=(dot(W,dobs)-dot(W,dsyn))**2
    r=sqrt(sum(delta)/len(delta))
    return r
    
def m2ssds(m):
    '''
    Split model vector into strike slip and dip slip components
    '''
    from numpy import arange
    i=arange(len(m)/2)
    iss=2*i
    ids=iss+1
    ss=m[iss]
    ds=m[ids]
    return ss,ds
