import h5py
import numpy as np
from numpy.linalg import inv, eigh
import matplotlib.pyplot as plt
import numba
import itertools
from wannierTB import *

@numba.jit(nopython=True)
def compute_Gkw(mu, ek_path, oms, eta, Sw, nimp):
    Gkw = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=np.complex128)
    for ik, ek in enumerate(ek_path):
        for iom, om in enumerate(oms):
            Gkw[ik,iom,:,:] = inv((om+1j*eta+mu)*np.eye(nimp)-ek-np.diag(Sw[iom,:]))
    return Gkw

import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))


np.set_printoptions(suppress=True,precision=10)
# grisb parameters
nimp = 6
cutoff = 0.0

# construct ek with semicircular DOS
Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')

# construct k-path
f = open('Sr2RuO4_wann3o_band.kpt','r')
Nkpath=int(f.readline())
print ('Nkpath=',Nkpath)
kx_path = np.zeros((Nkpath))
ky_path = np.zeros((Nkpath))
kz_path = np.zeros((Nkpath))
for i in range(Nkpath):
    tmp = f.readline().split()
    kx_path[i]=float(tmp[0])*2.0*np.pi
    ky_path[i]=float(tmp[1])*2.0*np.pi
    kz_path[i]=float(tmp[2])*2.0*np.pi
f.close()
#quit()
print ('kx_path=',kx_path)
print ('ky_path=',ky_path)
print ('kz_path=',kz_path)

eks_path_all = []
for i in range(len(kx_path)):
    kx, ky, kz = kx_path[i], ky_path[i], kz_path[i]
    #print(kx, ky, kz)
    tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,nimp//2,cutoff=cutoff)
    eks_path_all.append(tmp)
eks_path = np.array(eks_path_all)

U = 2.00
nfix = 4.0
eta = 0.0#1
#mu = -0.0
#mu = 4.0632
mu = 5.36
#mu = 5.111474
#mu = 5.1
#mu = 5.25 # triqs example
#data = np.loadtxt('Sw.inp').T
data = np.loadtxt('Sig.out').T
oms = data[0]
Sw = (data[1:7:2] + 1j*data[2:7:2]).T + 5.05
tmp_avg = (Sw[:,1] + Sw[:,2])/2.
Sw[:,1] = tmp_avg
Sw[:,2] = tmp_avg

Gf = compute_Gkw(mu, eks_path, oms, eta, Sw, nimp//2)
print(Gf.shape)

Nmid = 300
Z1 = 1./(1 - (Sw[Nmid,0]-Sw[Nmid-1,0]).real/(oms[Nmid]-oms[Nmid-1]) )
Z2 = 1./(1 - (Sw[Nmid,1]-Sw[Nmid-1,1]).real/(oms[Nmid]-oms[Nmid-1]) )
Z3 = 1./(1 - (Sw[Nmid,2]-Sw[Nmid-1,2]).real/(oms[Nmid]-oms[Nmid-1]) )
print('Z1=',Z1,'Z2=',Z2,'Z3=',Z3)
print(oms[Nmid],oms[Nmid-1])

plt.plot(oms,Sw[:,0].imag)
plt.plot(oms,Sw[:,1].imag)
plt.plot(oms,Sw[:,2].imag)
plt.show()

X, Y = np.meshgrid(np.arange(Nkpath),oms)
Z = np.zeros(X.shape)
Z = -2*( Gf[:,:,0,0] + Gf[:,:,1,1] + Gf[:,:,2,2] ).T.imag/np.pi

plt.title(r' $A(k,\omega)$',size=20)
im = plt.pcolormesh(X, Y, Z, cmap='afmhot', shading= 'gouraud', vmax=10)
plt.axhline(0,color='yellow',lw=0.5)
cbar = plt.colorbar(im)
#cbar.ax.set_title(r' $A(k,\omega)$',size=20)
cbar.ax.tick_params(labelsize=20)
plt.xlim(0,Nkpath-1)
plt.ylim(-3,1.5)
#plt.xlabel(r'$\mathbf{k}$',size=20)
plt.ylabel(r'$\omega$ (eV)',size=20)
plt.xticks([0,50,55,87,126,148],[r'$\Gamma$',r'$X$',r'$R$',r'$S$',r'$\Gamma$',r'$Z$'], size=20)
plt.yticks(size=20)
#plt.legend(loc='lower left', fontsize=20)
plt.text(1,1.0,'DMFT-CTQMC', color='yellow', size=20)

#plt.subplot(2,2,4)
#data_xy = np.loadtxt('gRISB/3o9b/U2.5J0.5/dos_xy.dat').T
#data_xz = np.loadtxt('gRISB/3o9b/U2.5J0.5/dos_xz.dat').T
#data_yz = np.loadtxt('gRISB/3o9b/U2.5J0.5/dos_yz.dat').T
##plt.title(r' $A(k,\omega)$',size=20)
#plt.plot(data_xy[0], data_xy[1], 'b-', label='Ru-$d_{xy}$')
#plt.plot(data_xz[0], data_xz[1], 'r-', label='Ru-$d_{xz}$')
#plt.plot(data_yz[0], data_yz[1], 'r-', label='Ru-$d_{yz}$')
#plt.xlim(-3,1.5)
#plt.ylim(0,2.0)
#plt.xlabel(r'$\omega$ (eV)',size=20)
#plt.ylabel('DOS (1/eV)',size=20)
#plt.xticks(Size=20)
#plt.yticks(size=20)
#plt.axvline(0,color='k',lw=0.8)
##plt.legend(loc='lower left', fontsize=20)
#plt.text(-2.75,1.75,'(d) gRISB', color='k', size=20)

#plt.subplots_adjust(hspace=0.45,wspace=0.3,left=0.1,right=0.9,bottom=0.1,top=0.9)
plt.tight_layout()
plt.savefig('SRO_dmft_ctqmc.png')
plt.show()
