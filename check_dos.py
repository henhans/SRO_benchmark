import numpy as np
from numpy.linalg import eigh, inv
from numba import jit
from wannierTB import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

no = 3
cutoff = 0.0
eta = 0.05

Rs, hmns, degs = parse_hopping_from_wannier90_hr_dat('Sr2RuO4_wann3o_hr.dat')
#print(hopp_dict[(-1,-1,-2)])
#print(hopp_dict)
#AssembleHk(0,0,0,hopp_dict)
#quit()

Nx = 50
Ny = 50
Nz = 10
Nom = 500
kxs = np.arange(0.0,2*np.pi,2*np.pi/Nx)
kys = np.arange(0.0,2*np.pi,2*np.pi/Ny)
kzs = np.arange(0.0,2*np.pi,2*np.pi/Nz)
oms = np.linspace(-4,3,Nom)

ek_list = []
for ikx, kx in enumerate(kxs):
    for iky, ky in enumerate(kys):
        for ikz, kz in enumerate(kzs):
            kxt = kx
            kyt = ky
            kzt = kz
            #kxt = (-kx+ky+kz)/2.
            #kyt = ( kx-ky+kz)/2.
            #kzt = ( kx+ky-kz)/2.
            tmp = AssembleHk(kxt,kyt,kzt,Rs,hmns,degs,no,cutoff=cutoff)
            #tmp = AssembleHk(kx,ky,kz,Rs,hmns,degs,no,cutoff=cutoff)
            ek_list.append(tmp)

#@jit(nopython=True)
#def compute_dos(Nom, oms, ek_list, eta):
#    dos = np.zeros((Nom))
#    for iom, om in enumerate(oms):
#        print ('iom=',iom)
#        for ek in ek_list:
#            evals, evecs = eigh(ek)
#            for e in evals:
#                dos[iom] += -np.imag(1./(om+1j*eta-e))/np.pi
#    return dos/len(ek_list)

@jit(nopython=True)
def compute_dos(Nom, oms, ek_list, eta, no):
    dos = np.zeros((Nom,no,no))
    for iom, om in enumerate(oms):
        print ('iom=',iom)
        for ek in ek_list:
            dos[iom] += -np.imag(inv( (om+1j*eta)*np.eye(no)-ek) )/np.pi
        dos[iom]= dos[iom]/len(ek_list)
    return dos

dos = compute_dos(Nom, oms, ek_list, eta, no) 

#data = np.loadtxt('../../../../dos/NiO/NiO.dos1ev').T

#plt.subplot(2,1,1)
plt.title('')
plt.plot(oms, dos[:,0,0], color='b', label='xy')
plt.plot(oms, dos[:,1,1], color='r', label='xz')
plt.plot(oms, dos[:,2,2], color='g', label='yz')

plt.legend(loc='best')

#plt.subplot(2,1,2)
#plt.plot(data[0], data[1], color='r')

plt.show()

np.savetxt('dos_xy.dat',np.vstack((oms,dos[:,0,0])).T)
np.savetxt('dos_xz.dat',np.vstack((oms,dos[:,1,1])).T)
np.savetxt('dos_yz.dat',np.vstack((oms,dos[:,2,2])).T)


