import h5py
import numpy as np
from numpy.linalg import inv, eigh
import matplotlib.pyplot as plt
import numba
import itertools

@numba.jit(nopython=True)
def compute_Gf_Sig(mu, ek_path, oms, eta, R, Lambda, eloc, nbath, nimp, ntot):
    #Gf = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
    #Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
    Gf = np.zeros((oms.shape[0],nimp,nimp),dtype=np.complex128)
    Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=np.complex128)
    for iom, om in enumerate(oms):
        print('iom=',iom)
        for ik, ek in enumerate(ek_path):
            Gf[iom,:,:] += R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
                               - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
            if ik == 0:
                Sig[iom,:,:] = ( (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc)[:nimp,:nimp] - inv(Gf[iom,:,:])[:nimp,:nimp]
        Gf[iom,:,:] = Gf[iom,:,:]/ek_path.shape[0]
    return Gf, Sig


#@numba.jit(nopython=True)
#def compute_Gf_Sig(ek_path, oms, eta, R, Lambda, nbath, nimp):
#    Gf = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=numba.complex128)
#    Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=numba.complex128)
#    for ik, ek in enumerate(ek_path):
#        for iom, om in enumerate(oms):
#            Gf[ik,iom,:,:] = R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
#                              - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
#            if ik == 10:
#                Sig[iom,:,:] = om + 1j*eta - ek - inv(Gf[ik,iom])
#    return Gf, Sig

def compute_Aekw(ek, om, eta, R, Lambda, nbath, nimp):
    Gf = np.zeros((nimp,nimp),dtype=np.complex128)
    Gf = R.conj().T.dot( inv( (om+1j*eta)*np.eye(nbath)
                 - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
    return -np.trace(Gf).imag/np.pi

from wannierTB import *

np.set_printoptions(suppress=True,precision=10)
cutoff = 0.0
# grisb parameters
ntot = 36
nimp = 6
nbath= 30

# construct ek with semicircular DOS 
Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')
#print(hopp_dict[(-1,-1,-2)])
#print(hopp_dict)
#AssembleHk(0,0,0,hopp_dict)
#quit()

Nx = 40
Ny = 40
Nz = 10
kxs = np.arange(0.0,2*np.pi,2*np.pi/Nx)
kys = np.arange(0.0,2*np.pi,2*np.pi/Ny)
kzs = np.arange(0.0,2*np.pi,2*np.pi/Nz)

ek_all_list = []
for ikx, kx in enumerate(kxs):
    for iky, ky in enumerate(kys):
        for ikz, kz in enumerate(kzs):
            kxt = (-kx+ky+kz)/2.
            kyt = ( kx-ky+kz)/2.
            kzt = ( kx+ky-kz)/2.
            tmp = AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,nimp//2,cutoff=cutoff)
            #tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,nimp//2,cutoff=cutoff)
            tmp = np.kron(tmp,np.eye(2))
            ek_all_list.append(tmp)
# compute local energy and subtract it
eloc_all = sum(ek_all_list)/len(ek_all_list)
eloc = np.zeros(eloc_all.shape, dtype=eloc_all.dtype)
eloc[:nimp,:nimp] = eloc_all[:nimp,:nimp]
print('eloc_=')
print(eloc[:nimp:2,:nimp:2])
#quit()
# subtract eloc for correlated subspace
eks = []
for ek in ek_all_list:
    eks.append(ek-eloc)
eks = np.array(eks)

Us = np.arange(0.1,3.55,0.1)
Nom = 200
oms = np.linspace(-4.0,3.0,Nom)
eta = 0.05#0.00005
#oms = np.linspace(-0.01,0.01,Nom)
#eta = 0.00005

#fo = open('U_Z.dat','w')
for U in [1.5]:#Us:
    fh5 = h5py.File('sols.h5','r')
    R = fh5['U%.2f/R'%(U)][...]
    Lambda = fh5['U%.2f/Lambda'%(U)][...]
    mu = fh5['U%.2f/mu'%(U)][...]
    muDC = 0.0#fh5['U%.2f/muDC'%(U)][...]
    #Sig = fh5['U%.2f/Sig'%(U)][...]
    fh5.close()

    Gf, Sig = compute_Gf_Sig(mu, eks, oms, eta, R, Lambda, eloc, nbath, nimp, ntot)
    #Gf, Sig = compute_Gf_Sig(eks, oms, eta, R, Lambda, nbath, nimp)
    print(Sig.shape, Gf.shape)

    Z1 = 1./(1 - (Sig[Nom//2,0,0]-Sig[Nom//2-1,0,0]).real/(oms[Nom//2]-oms[Nom//2-1]) )
    print('Z1=',Z1)

    Aw = -Gf.imag/np.pi
    print(Aw.shape)
    #print(simps(Aw,oms))

    plt.plot(oms, Aw[:,4,4] + Aw[:,5,5], 'g-', label='Ru-yz')
    plt.plot(oms, Aw[:,2,2] + Aw[:,3,3], 'r-', label='Ru-xz')
    plt.plot(oms, Aw[:,0,0] + Aw[:,1,1], 'b-', label='Ru-xy')
    plt.axvline(0,color='k',ls='-')
    plt.ylim(0,)
    plt.xlim(-4,3)
    plt.xlabel('$\omega$ (eV)',size=15)
    plt.ylabel('DOS (1/eV)',size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('DOS.pdf')
    plt.show()

    np.savetxt('dos_xy.dat', np.vstack((oms, Aw[:,0,0] + Aw[:,1,1] )).T)
    np.savetxt('dos_xz.dat', np.vstack((oms, Aw[:,2,2] + Aw[:,3,3] )).T)
    np.savetxt('dos_yz.dat', np.vstack((oms, Aw[:,4,4] + Aw[:,5,5] )).T)

    #np.savetxt('Aw_U%.2fn%.2f_9b.dat'%(U,nfill),np.vstack((oms,Aw)).T)
    #print(U, Z1, file=fo)

#fo.close()
