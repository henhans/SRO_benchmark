import numpy as np
from io import StringIO
from collections import OrderedDict
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib
from wannierTB import *
matplotlib.use('TKAgg')

Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')
#print(hopp_dict[(-1,-1,-2)])
#print(hopp_dict)
#AssembleHk(0,0,0,hopp_dict)
#quit()

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

epath_U0 = np.zeros((3,Nkpath))
evecb_U0 = np.zeros((3,3,Nkpath),dtype=np.complex128)
#epath = zeros((Nkpath))
for i in range(Nkpath):
    kx, ky, kz = kx_path[i], ky_path[i], kz_path[i]
    #print(kx, ky, kz)

    ek = AssembleHk(kx,ky,kz,Rs,hmns,degs,3) + 0.*np.eye(3) # shift EF
    #ekqp = R.dot( (get_ek(kx, ky, kz))-eloc ).dot(R.conj().T) + Lam

    epath_U0[:,i], evecb_U0[:,:,i] = eigh(ek)#[0]
    #epath[:,i] = eigh(ekqp)[0]

plt.figure(figsize=(8,6))

alpha = 0.5
scale = 50

x = np.arange(Nkpath)

plt.subplot(1,2,1)

plt.plot(epath_U0[0,:],'-', color='k')
plt.plot(epath_U0[1,:],'-', color='k')
plt.plot(epath_U0[2,:],'-', color='k')

plt.scatter(x, epath_U0[0,:], s=scale*(np.abs(evecb_U0[0,0,:])**2), c='b', alpha=alpha, label='Ru-$d_{xy}$')
plt.scatter(x, epath_U0[1,:], s=scale*(np.abs(evecb_U0[0,1,:])**2), c='b', alpha=alpha)
plt.scatter(x, epath_U0[2,:], s=scale*(np.abs(evecb_U0[0,2,:])**2), c='b', alpha=alpha)

plt.scatter(x, epath_U0[0,:], s=scale*(np.abs(evecb_U0[1,0,:])**2), c='r', alpha=alpha, label='Ru-$d_{xz}$')
plt.scatter(x, epath_U0[1,:], s=scale*(np.abs(evecb_U0[1,1,:])**2), c='r', alpha=alpha)
plt.scatter(x, epath_U0[2,:], s=scale*(np.abs(evecb_U0[1,2,:])**2), c='r', alpha=alpha)

plt.scatter(x, epath_U0[0,:], s=scale*(np.abs(evecb_U0[2,0,:])**2), c='r', alpha=alpha, label='Ru-$d_{yz}$')
plt.scatter(x, epath_U0[1,:], s=scale*(np.abs(evecb_U0[2,1,:])**2), c='r', alpha=alpha)
plt.scatter(x, epath_U0[2,:], s=scale*(np.abs(evecb_U0[2,2,:])**2), c='r', alpha=alpha)

plt.axhline(0,ls='-',color='k',lw=0.8)

plt.ylabel('$E$ (eV)', size=15)
plt.xlim(0,Nkpath)
plt.ylim(-3.,1.5)
plt.xticks([0,50,100,171,187],[r'$\Gamma$',r'$M$  ',r'  $X$',r'$\Gamma$',r'$Z$'], size=15)
#plt.xticks(size=15)
plt.yticks(size=15)
lgnd = plt.legend(loc="lower center", scatterpoints=1, fontsize=15)
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])
    handle.set_alpha(1.0)

plt.subplot(1,2,2)

data_eg = np.loadtxt('dos_xy.dat').T
data_t2g = np.loadtxt('dos_xz.dat').T
data_p = np.loadtxt('dos_yz.dat').T

plt.plot(data_eg[0], data_eg[1], 'b-', label='Ru-$d_{xy}$')
plt.plot(data_t2g[0], data_t2g[1], 'r-', label='Ru-$d_{xz}$')
plt.plot(data_p[0], data_p[1], 'r-', label='Ru-$d_{yz}$')

plt.xlim(-3.,1.5)
plt.ylim(0,1.5)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('$\omega$ (eV)',size=15)
plt.ylabel('DOS (1/eV)',size=15)

plt.axvline(0,ls='-',color='k',lw=0.8)

plt.legend(loc='upper right',fontsize=15)

#plt.ylim(-3.2,7.2)
#plt.yticks([-2,0,4,6],["","","",""],size=15)
#plt.xticks(size=15)
#plt.yticks(size=15)
#lgnd = plt.legend(loc="lower right", scatterpoints=1, fontsize=12)
#for handle in lgnd.legendHandles:
#    handle.set_sizes([20.0])
#    handle.set_alpha(1.0)


plt.tight_layout()
plt.savefig('WannTB_char_SRO.pdf')
plt.show()

