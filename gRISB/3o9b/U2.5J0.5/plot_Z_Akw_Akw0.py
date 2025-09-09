import h5py
import numpy as np
from numpy.linalg import inv, eigh
import matplotlib.pyplot as plt
import numba
import itertools
from wannierTB import parse_hopping_from_wannier90_hr_dat, AssembleHk

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
                Sig[iom,:,:] = (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc - inv(Gf[ik,iom])
    return Gf, Sig

def compute_Aekw(ek, om, eta, R, Lambda, nbath, nimp):
    Gf = np.zeros((nimp,nimp),dtype=np.complex128)
    Gf = R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
                 - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
    return -np.trace(Gf).imag/np.pi

import matplotlib.pyplot as plt


plt.figure(figsize=(8,15))

np.set_printoptions(suppress=True,precision=10)
# grisb parameters
ntot = 24
nimp = 6
nbath= 18
cutoff = 0.0

# construct ek with semicircular DOS
Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')

Nx = 40
Ny = 40
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

# construct k-path
Nksec= 20
Nkpath = 3*Nksec+1
kx_path = np.hstack( (np.arange(0,np.pi,np.pi/Nksec), np.arange(0.0,np.pi,np.pi/Nksec),
                         np.arange(np.pi,-np.pi/Nksec, -np.pi/Nksec) ) )
ky_path = np.hstack( (np.zeros(Nksec), np.ones(Nksec)*np.pi, np.arange(np.pi,-np.pi/Nksec, -np.pi/Nksec) ) )
print('kx_path',kx_path)
print('ky_path',ky_path)
epath_U0 = np.zeros((nimp,Nkpath))
eks_path_all = []
for i in range(3*Nksec+1):
    kx, ky, kz = kx_path[i], ky_path[i], 0.#kz_path[i]
    #print(kx, ky, kz)
    kxt = (-kx+ky+kz)/2.
    kyt = ( kx-ky+kz)/2.
    kzt = ( kx+ky-kz)/2.
    tmp = AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    #tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    tmp = np.kron(tmp,np.eye(2))
    eks_path_all.append(tmp)

# construct and subtract eloc for grisb
eloc = sum(eks_all)/len(eks_all)
eks = []
eks_path = []
for e in eks_all:
    eks.append(e-eloc)
eks = np.array(eks)
for e in eks_path_all:
    eks_path.append(e-eloc)
eks_path = np.array(eks_path)
print('eloc=')
print(eloc)

Nom = 500
oms = np.linspace(-4,4,Nom)
eta = 0.05#0.00005

U = 2.5
fh5 = h5py.File('sols.h5','r')
R = fh5['U%.2f/R'%(U)][...]
Lambda = fh5['U%.2f/Lambda'%(U)][...]
#Sig = fh5['U%.2f/Sig'%(U)][...]
mu = fh5['U%.2f/mu'%(U)][...]
fh5.close()

Gf, Sig = compute_Gf_Sig(mu, eks_path, oms, eta, R, Lambda, eloc, nbath, nimp, ntot)
print(Sig.shape, Gf.shape)

Z1 = 1./(1 - (Sig[Nom//2,0,0]-Sig[Nom//2-1,0,0]).real/(oms[Nom//2]-oms[Nom//2-1]) )
Z2 = 1./(1 - (Sig[Nom//2,2,2]-Sig[Nom//2-1,2,2]).real/(oms[Nom//2]-oms[Nom//2-1]) )
Z3 = 1./(1 - (Sig[Nom//2,4,4]-Sig[Nom//2-1,4,4]).real/(oms[Nom//2]-oms[Nom//2-1]) )
print('Z1=',Z1,'Z2=',Z2,'Z3=',Z3)

#plt.plot(oms,Sig[:,0,0])
#plt.plot(oms,Sig[:,2,2])
#plt.plot(oms,Sig[:,4,4])
#plt.show()

X, Y = np.meshgrid(np.arange(Nkpath),oms)
Z = np.zeros(X.shape)
Z = -( Gf[:,:,0,0] + Gf[:,:,1,1] + Gf[:,:,2,2] + Gf[:,:,3,3] + Gf[:,:,4,4] + Gf[:,:,5,5] ).T.imag/np.pi

plt.subplot(2,1,1)
plt.title(r' $A(k,\omega)$',size=15)
im = plt.pcolormesh(X, Y, Z, cmap='afmhot', shading= 'gouraud', vmax=10)
cbar = plt.colorbar(im)
#cbar.ax.set_title(r' $A(k,\omega)$',size=20)
cbar.ax.tick_params(labelsize=15)
plt.xlim(0,Nkpath)
plt.ylim(-3,1)
plt.xlabel(r'$\mathbf{k}$',size=15)
plt.ylabel(r'$\omega$',size=15)
plt.xticks([0,20,40,60],[r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"],size=15)
plt.yticks(size=15)
#plt.legend(loc='lower left', fontsize=15)
plt.text(1,0.5,'(a) gRISB', color='yellow', size=15)

# construct ek for 2D drawing
Nx_plt = 128+1
Ny_plt = 128+1
kxs_plt = np.arange(-np.pi,np.pi+2*np.pi/(Nx_plt-1),2*np.pi/(Nx_plt-1))
kys_plt = np.arange(-np.pi,np.pi+2*np.pi/(Ny_plt-1),2*np.pi/(Ny_plt-1))
# compute Akw0
Akw0 = np.zeros((Nx_plt,Ny_plt))
for ikx, iky in itertools.product(range(Nx_plt),range(Ny_plt)):
    kx, ky, kz = kxs_plt[ikx], kys_plt[iky], 0.
    #print(kx, ky, kz)
    kxt = (-kx+ky+kz)/2.
    kyt = ( kx-ky+kz)/2.
    kzt = ( kx+ky-kz)/2.
    tmp = AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    #tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    tmp = np.kron(tmp,np.eye(2))
    ek = tmp - eloc
    Akw0[ikx,iky] = compute_Aekw(ek, 0, 0.005, R, Lambda, nbath, nimp)

X, Y = np.meshgrid(kxs_plt,kys_plt)
plt.subplot(2,1,2)
plt.title(r' $A(k,\omega=0)$',size=15)
im = plt.pcolormesh(X, Y, Akw0, cmap='afmhot', vmax=40)#, shading= 'gouraud')
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
plt.text(-1.2,0,'(b) gRISB', color='yellow', size=15)

plt.subplots_adjust(hspace=0.45,wspace=0.3,left=0.1,right=0.9,bottom=0.1,top=0.9)
#plt.tight_layout()
plt.savefig('Akw_SRO.png')
plt.show()
