import h5py
import numpy as np
from numpy.linalg import inv, eigh
import matplotlib.pyplot as plt
import numba
import itertools
from wannierTB import *

@numba.jit(nopython=True)
def compute_Gf_Sig(mu, ek_path, oms, eta, R, Lambda, eloc, nbath, nimp, ntot):
    #Gf = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
    #Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
    Gf = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=np.complex128)
    Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=np.complex128)
    #for iom, om in enumerate(oms):
    #    print('iom=',iom)
    #    for ik, ek in enumerate(ek_path):
    #        Gf[iom,:,:] += R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
    #                           - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
    #        if ik == 0:
    #            Sig[iom,:,:] = ( (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc)[:nimp,:nimp] - inv(Gf[iom,:,:])[:nimp,:nimp]
    #    Gf[iom,:,:] = Gf[iom,:,:]/ek_path.shape[0]
    for ik, ek in enumerate(ek_path):
        for iom, om in enumerate(oms):
            Gf[ik,iom,:,:] = R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
                              - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
            if ik == 10:
                Sig[iom,:,:] = (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc - inv(Gf[ik,iom,:,:])
    return Gf, Sig

@numba.jit(nopython=True)
def compute_Gkw(mu, ek_path, oms, eta, Sw, nimp):
    Gkw = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=np.complex128)
    for ik, ek in enumerate(ek_path):
        for iom, om in enumerate(oms):
            Gkw[ik,iom,:,:] = inv((om+1j*eta+mu)*np.eye(nimp)-ek[::2,::2]-np.diag(Sw[iom,:]))
    return Gkw

#@numba.jit(nopython=True)
#def compute_Gf_Sig(mu, ek_path, oms, eta, Rdp, Lambdadp, eloc, nbath, nimp, ndp, n_p, ntot):
#    #Gf = np.zeros((oms.shape[0],ndp,ndp),dtype=numba.complex128)
#    #Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
#    Gf = np.zeros((ek_path.shape[0],oms.shape[0],ndp,ndp),dtype=np.complex128)
#    Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=np.complex128)
#    for ik, ek in enumerate(ek_path):
#        for iom, om in enumerate(oms):
#            print('iom=',iom)
#            #print(Rdp.shape)
#            #print(ek.shape)
#            Gf[ik,iom,:,:] = Rdp.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath+n_p)
#                               - Rdp.dot(ek).dot(Rdp.conj().T) - Lambdadp ) ).dot(Rdp)
#            if ik == 0:
#                Sig[iom,:,:] = ( (om + 1j*eta + mu)*np.eye(ndp) - ek - eloc)[:nimp,:nimp] - inv(Gf[ik,iom,:,:])[:nimp,:nimp]
#    return Gf, Sig

def compute_Aekw(ek, om, eta, R, Lambda, nbath, nimp):
    Gf = np.zeros((nimp,nimp),dtype=np.complex128)
    Gf = R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
                 - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
    return -np.trace(Gf).imag/np.pi


def compute_Aekw_dmft(mu, ek, om, eta, Sw, nimp):
    Gf = np.zeros((nimp//2,nimp//2),dtype=np.complex128)
    Gf = inv((om+1j*eta+mu)*np.eye(nimp//2)-ek[::2,::2]-(np.diag(Sw[299,:]+Sw[300,:])/2.))
    return -2*np.trace(Gf).imag/np.pi

import matplotlib.pyplot as plt


plt.figure(figsize=(8,3.5))


np.set_printoptions(suppress=True,precision=10)
# grisb parameters
nimp = 6
cutoff = 0.0

# construct ek with semicircular DOS
Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')

Nx = 20
Ny = 20
Nz = 10
kxs = np.arange(0.0,2*np.pi,2*np.pi/Nx)
kys = np.arange(0.0,2*np.pi,2*np.pi/Ny)
kzs = np.arange(0.0,2*np.pi,2*np.pi/Nz)

eks_all = []
eband = np.zeros((nimp,nimp,Nx,Ny,Nz))
for ikx, iky, ikz in itertools.product(range(Nx),range(Ny),range(Nz)):
    kx, ky, kz = kxs[ikx], kys[iky], kzs[ikz]
    kxt = (-kx+ky+kz)/2.
    kyt = ( kx-ky+kz)/2.
    kzt = ( kx+ky-kz)/2.
    tmp = AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    #tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    tmp = np.kron(tmp,np.eye(2))
    evals, evecs = eigh(tmp)
    eband[:,:,ikx,iky,ikz] = evecs.conj().T.dot(tmp).dot(evecs)
    eks_all.append(tmp)

# construct and subtract eloc for grisb
eloc_all = sum(eks_all)/len(eks_all)
eloc = np.zeros(eloc_all.shape, dtype=eloc_all.dtype)
eloc[:nimp,:nimp] = eloc_all[:nimp,:nimp]
print('eloc_=')
print(eloc[:nimp:2,:nimp:2])
eks = []
eks_path = []
for e in eks_all:
    eks.append(e-eloc)
eks = np.array(eks)
print('eloc=')
print(eloc)

##########################################################
# RISB
##########################################################

##Nom = 500
##oms = np.linspace(-10,6,Nom)
#eta = 0.05#0.00005
#
#nbath= 6
#ntot = nimp + nbath
#
#U = 2.50
#fh5 = h5py.File('gRISB/3o3b/JoverU0.2/sols.h5','r')
#R = fh5['U%.2f/R'%(U)][...]
#Lambda = fh5['U%.2f/Lambda'%(U)][...]
##Sig = fh5['U%.2f/Sig'%(U)][...]
#mu = fh5['U%.2f/mu'%(U)][...]
##muDC = fh5['U%.2f/muDC'%(U)][...]
#fh5.close()
#
## construct ek for 2D drawing
#Nx_plt = 101#128+1
#Ny_plt = 101#128+1
#kxs_plt = np.arange(-np.pi,np.pi+2*np.pi/(Nx_plt-1),2*np.pi/(Nx_plt-1))
#kys_plt = np.arange(-np.pi,np.pi+2*np.pi/(Ny_plt-1),2*np.pi/(Ny_plt-1))
## compute Akw0
#Akw0 = np.zeros((Nx_plt,Ny_plt))
#for ikx, iky in itertools.product(range(Nx_plt),range(Ny_plt)):
#    kx, ky, kz = kxs_plt[ikx], kys_plt[iky], 0
#    kxt = (-kx+ky+kz)/2.
#    kyt = ( kx-ky+kz)/2.
#    kzt = ( kx+ky-kz)/2.
#    ek = np.kron(AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff),np.eye(2))
#    ek-= eloc
#    Akw0[ikx,iky] = compute_Aekw(ek, 0, eta, R, Lambda, nbath, nimp)
#
#X, Y = np.meshgrid(kxs_plt,kys_plt)
##plt.subplot(3,2,5)
#plt.title(r' $A(k,\omega=0)$',size=15)
#im = plt.pcolormesh(X, Y, Akw0, cmap='afmhot', vmax=40)#, shading= 'gouraud')
#cbar = plt.colorbar(im)
##cbar.ax.set_title(r' $A(k,\omega=0)$',size=20)
#cbar.ax.tick_params(labelsize=15)
#plt.xlim(-np.pi,np.pi)
#plt.ylim(-np.pi,np.pi)
#plt.xlabel(r'$\mathbf{k}_x$',size=15)
#plt.ylabel(r'$\mathbf{k}_y$',size=15)
#plt.xticks(size=15)
#plt.yticks(size=15)
##plt.legend(loc='lower left', fontsize=15)
#plt.text(-1.2,0,'(e) gRISB', color='yellow', size=15)
#plt.show()
#quit()

##########################################################################
# g-RISB
##########################################################################

nbath= 3*6
ntot = nimp + nbath

U = 2.30
#fh5 = h5py.File('gRISB/3o9b/U2.5J0.5/sols.h5','r')
fh5 = h5py.File('gRISB/3o9b/U2.3/sols.h5','r')
R = fh5['U%.2f/R'%(U)][...]
Lambda = fh5['U%.2f/Lambda'%(U)][...]
#Sig = fh5['U%.2f/Sig'%(U)][...]
mu = fh5['U%.2f/mu'%(U)][...]
#muDC = fh5['U%.2f/muDC'%(U)][...]
fh5.close()
eta = 0.005

# construct ek for 2D drawing
Nx_plt = 101#128+1
Ny_plt = 101#128+1
kxs_plt = np.arange(-np.pi,np.pi+2*np.pi/(Nx_plt-1),2*np.pi/(Nx_plt-1))
kys_plt = np.arange(-np.pi,np.pi+2*np.pi/(Ny_plt-1),2*np.pi/(Ny_plt-1))
# compute Akw0
Akw0 = np.zeros((Nx_plt,Ny_plt))
for ikx, iky in itertools.product(range(Nx_plt),range(Ny_plt)):
    kx, ky, kz = kxs_plt[ikx], kys_plt[iky], 0
    kxt = (-kx+ky+kz)/2.
    kyt = ( kx-ky+kz)/2.
    kzt = ( kx+ky-kz)/2.
    ek = np.kron(AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff),np.eye(2))
    ek-= eloc
    Akw0[ikx,iky] = compute_Aekw(ek, 0, eta, R, Lambda, nbath, nimp)

X, Y = np.meshgrid(kxs_plt,kys_plt)
plt.subplot(1,2,1)
plt.title(r' $A(k,\omega=0)$',size=15)
im = plt.pcolormesh(X, Y, Akw0, cmap='afmhot', vmax=30)#, shading= 'gouraud')
cbar = plt.colorbar(im)
#cbar.ax.set_title(r' $A(k,\omega=0)$',size=20)
cbar.ax.tick_params(labelsize=15)
plt.xlim(-np.pi,np.pi)
plt.ylim(-np.pi,np.pi)
plt.xlabel(r'$\mathbf{k}_x$',size=15)
plt.ylabel(r'$\mathbf{k}_y$',size=15)
plt.xticks(size=15)
plt.yticks(size=15)
#plt.legend(loc='lower left', fontsize=15)
plt.text(-1.2,0,'(a) g-RISB', color='yellow', size=15)
#plt.show()


################################################

data = np.loadtxt('ctqmc/Sig.out').T
oms = data[0]
Sw = (data[1:7:2] + 1j*data[2:7:2]).T + 5.05
tmp_avg = (Sw[:,1] + Sw[:,2])/2.
Sw[:,1] = tmp_avg
Sw[:,2] = tmp_avg
mu = 5.36
eta = 0.005
print(oms[300])

# construct ek for 2D drawing
Nx_plt = 101#128+1
Ny_plt = 101#128+1
kxs_plt = np.arange(-np.pi,np.pi+2*np.pi/(Nx_plt-1),2*np.pi/(Nx_plt-1))
kys_plt = np.arange(-np.pi,np.pi+2*np.pi/(Ny_plt-1),2*np.pi/(Ny_plt-1))
# compute Akw0
Akw0 = np.zeros((Nx_plt,Ny_plt))
for ikx, iky in itertools.product(range(Nx_plt),range(Ny_plt)):
    kx, ky, kz = kxs_plt[ikx], kys_plt[iky], 0
    kxt = (-kx+ky+kz)/2.
    kyt = ( kx-ky+kz)/2.
    kzt = ( kx+ky-kz)/2.
    ek = np.kron(AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff),np.eye(2))
    #ek-= eloc
    Akw0[ikx,iky] = compute_Aekw_dmft(mu, ek, 0, eta, Sw, nimp)

X, Y = np.meshgrid(kxs_plt,kys_plt)
plt.subplot(1,2,2)
plt.title(r' $A(k,\omega=0)$',size=15)
im = plt.pcolormesh(X, Y, Akw0, cmap='afmhot', vmax=30)#, shading= 'gouraud')
cbar = plt.colorbar(im)
#cbar.ax.set_title(r' $A(k,\omega=0)$',size=20)
cbar.ax.tick_params(labelsize=15)
plt.xlim(-np.pi,np.pi)
plt.ylim(-np.pi,np.pi)
plt.xlabel(r'$\mathbf{k}_x$',size=15)
plt.ylabel(r'$\mathbf{k}_y$',size=15)
plt.xticks(size=15)
plt.yticks(size=15)
#plt.legend(loc='lower left', fontsize=15)
plt.text(-2.2,0,'(b) DMFT-CTQMC', color='yellow', size=15)

#Gf = compute_Gkw(mu, eks_path, oms, eta, Sw, nimp//2)
##print(Gf.shape)
#
#X, Y = np.meshgrid(np.arange(Nkpath),oms)
#Z = np.zeros(X.shape)
#Z = -2*( Gf[:,:,0,0] + Gf[:,:,1,1] + Gf[:,:,2,2] ).T.imag/np.pi
#
##plt.title(r' $A(k,\omega)$',size=20)
#im = plt.pcolormesh(X, Y, Z, cmap='afmhot', shading= 'gouraud', vmax=10)
#plt.axhline(0,color='yellow',lw=0.5)
#cbar = plt.colorbar(im)
##cbar.ax.set_title(r' $A(k,\omega)$',size=20)
#cbar.ax.tick_params(labelsize=15)
#plt.xlim(0,Nkpath-1)
#plt.ylim(-3,2.0)
##plt.xlabel(r'$\mathbf{k}$',size=20)
#plt.ylabel(r'$\omega$ (eV)',size=20)
#plt.xticks([0,50,55,87,126,148],[r'$\Gamma$',r'$X$',r'$R$',r'$S$',r'$\Gamma$',r'$Z$'], size=15)
#plt.yticks(size=15)
##plt.legend(loc='lower left', fontsize=20)
#plt.text(1,1.5,'(e) DMFT(CTQMC)', color='yellow', size=15)

#plt.subplots_adjust(hspace=0.45,wspace=0.3,left=0.1,right=0.9,bottom=0.1,top=0.9)
plt.tight_layout()
plt.savefig('SRO_FS.png')
plt.show()
