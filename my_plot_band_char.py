import numpy as np
from io import StringIO
from collections import OrderedDict
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib
from numba import jit
matplotlib.use('TKAgg')

########################################################################
#                  Wannier tight-binding tools                         #
########################################################################
#@jit(nopython=True)
#def AssembleHk(kx,ky,kz,hopp_dict, cutoff=0.005):
#    """This is tight-binding Hamiltonian for the two-band model
#    with hopping parameters listed in "wannier_hr.dat".
#    """
#    Hk = np.zeros((3,3), dtype=complex)
#    for R, hmn in hopp_dict.items():
#        #print('R=',R)
#        for m in range(3):
#            for n in range(3):
#                if np.abs(hmn["h"][m,n])/float(hmn["deg"]) > cutoff:
#                    Hk[m,n] += hmn["h"][m,n]*np.exp(1j*(kx*R[0]+ky*R[1]+kz*R[2]))/float(hmn["deg"])
#    return Hk
@jit(nopython=True)
def AssembleHk(kx,ky,kz,Rs,hmns,degs, no, cutoff=0.00):
    """This is tight-binding Hamiltonian for the two-band model
    with hopping parameters listed in "wannier_hr.dat".
    """
    Hk = np.zeros((no,no), dtype=np.complex128)#numba.complex128)
    for i in range(Rs.shape[0]):
        #print('R=',R)
        for m in range(no):
            for n in range(no):
                if np.abs(hmns[i][m,n])/float(degs[i]) > cutoff:
                    Hk[m,n] += hmns[i][m,n]*np.exp(1j*(kx*Rs[i][0]+ky*Rs[i][1]+kz*Rs[i][2]))/float(degs[i])
    return Hk


def parse_hopping_from_wannier90_hr_dat(filename):
    # read in hamiltonian matrix, in eV
    f=open(filename,"r")
    ln=f.readlines()
    f.close()
    #
    # get number of wannier functions
    num_wan=int(ln[1])
    # get number of Wigner-Seitz points
    num_ws=int(ln[2])
    # get degenereacies of Wigner-Seitz points
    deg_ws=[]
    for j in range(3,len(ln)):
        sp=ln[j].split()
        for s in sp:
            deg_ws.append(int(s))
        if len(deg_ws)==num_ws:
            last_j=j
            break
        if len(deg_ws)>num_ws:
            raise Exception("Too many degeneracies for WS points!")
    deg_ws=np.array(deg_ws,dtype=int)
    # now read in matrix elements
    # Convention used in w90 is to write out:
    # R1, R2, R3, i, j, ham_r(i,j,R)
    # where ham_r(i,j,R) corresponds to matrix element < i | H | j+R >
    ham_r={} # format is ham_r[(R1,R2,R3)]["h"][i,j] for < i | H | j+R >
    ind_R=0 # which R vector in line is this?
    for j in range(last_j+1,len(ln)):
        sp=ln[j].split()
        # get reduced lattice vector components
        ham_R1=int(sp[0])
        ham_R2=int(sp[1])
        ham_R3=int(sp[2])
        # get Wannier indices
        ham_i=int(sp[3])-1
        ham_j=int(sp[4])-1
        # get matrix element
        ham_val=float(sp[5])+1.0j*float(sp[6])
        # store stuff, for each R store hamiltonian and degeneracy
        ham_key=(ham_R1,ham_R2,ham_R3)
        if (ham_key in ham_r)==False:
            ham_r[ham_key]={
                "h":np.zeros((num_wan,num_wan),dtype=complex),
                "deg":deg_ws[ind_R]
                }
            ind_R+=1
        ham_r[ham_key]["h"][ham_i,ham_j]=ham_val

    Rs = []
    hmns = []
    degs = []
    for R, hmn in ham_r.items():
        #print('R=',R)
        Rs.append(R) 
        hmns.append(hmn["h"])
        degs.append(hmn["deg"])
    return np.array(Rs), np.array(hmns), np.array(degs)
    #return ham_r, num_wan


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

no = 3

epath_U0 = np.zeros((no,Nkpath))
evecb_U0 = np.zeros((no,no,Nkpath),dtype=np.complex128)
#epath = zeros((Nkpath))
for i in range(Nkpath):
    kx, ky, kz = kx_path[i], ky_path[i], kz_path[i]
    #print(kx, ky, kz)

    ek = AssembleHk(kx,ky,kz,Rs,hmns,degs,no) + 0.*np.eye(no) # shift EF
    #ekqp = R.dot( (get_ek(kx, ky, kz))-eloc ).dot(R.conj().T) + Lam

    epath_U0[:,i], evecb_U0[:,:,i] = eigh(ek)#[0]
    #epath[:,i] = eigh(ekqp)[0]

plt.figure(figsize=(8,6))

alpha = 0.5
scale = 50

x = np.arange(Nkpath)

#plt.subplot(3,1,1)

for i in range(no):
    plt.plot(epath_U0[i,:],'-', color='k')

plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[0,0,:])**2, c='b', alpha=alpha, label='Ru-$d_{xy}$')
for i in range(1,no):
    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[0,i,:])**2, c='b', alpha=alpha)

plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[1,0,:])**2, c='r', alpha=alpha, label='Ru-$d_{xz}$')
for i in range(1,no):
    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[1,i,:])**2, c='r', alpha=alpha)

plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[2,0,:])**2, c='g', alpha=alpha, label='Ru-$d_{yz}$')
for i in range(1,no):
    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[2,i,:])**2, c='g', alpha=alpha)

plt.axhline(0,ls='-',color='k',lw=0.8)

plt.ylabel('$E$ (eV)', size=15)
plt.xlim(0,Nkpath)
plt.ylim(-3.,1.)
#plt.xticks([0,scale,100,171,230,280,330,401],[r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$'], size=15)
#plt.yticks(size=15)
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=12)
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])
    handle.set_alpha(1.0)

#plt.subplot(3,1,2)
#
#for i in range(no):
#    plt.plot(epath_U0[i,:],'-', color='k')
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[5,0,:])**2, c='b', alpha=alpha, label='Fe2-$d_{xy}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[5,i,:])**2, c='b', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[8,0,:])**2, c='r', alpha=alpha, label='Fe2-$d_{xz}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[8,i,:])**2, c='r', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[6,0,:])**2, c='g', alpha=alpha, label='Fe2-$d_{yz}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[6,i,:])**2, c='g', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[9,0,:])**2, c='c', alpha=alpha, label='Fe2-$d_{x^2-y^2}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[9,i,:])**2, c='c', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[7,0,:])**2, c='m', alpha=alpha, label='Fe2-$d_{z^2}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[7,i,:])**2, c='m', alpha=alpha)
#
#plt.axhline(0,ls='-',color='k',lw=0.8)
#
##plt.ylabel('$E$ (eV)', size=15)
#plt.xlim(0,Nkpath)
#plt.ylim(-4.6,3.1)
##plt.xticks([0,scale,100,171,230,280,330,401],[r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$'], size=15)
##plt.yticks([-2,0,4,6],["","","",""],size=15)
#lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=12)
#for handle in lgnd.legendHandles:
#    handle.set_sizes([20.0])
#    handle.set_alpha(1.0)
#
#plt.subplot(3,1,3)
#
#for i in range(no):
#    plt.plot(epath_U0[i,:],'-', color='k')
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[10,0,:])**2, c='b', alpha=alpha, label='As1-$p_{x}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[10,i,:])**2, c='b', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[11,0,:])**2, c='r', alpha=alpha, label='As1-$p_{y}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[11,i,:])**2, c='r', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[12,0,:])**2, c='g', alpha=alpha, label='As1-$p_{z}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[12,i,:])**2, c='g', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[13,0,:])**2, c='c', alpha=alpha, label='As2-$p_{x}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[13,i,:])**2, c='c', alpha=alpha)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[14,0,:])**2, c='m', alpha=alpha, label='As2-$p_{y}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[14,i,:])**2, c='m', alpha=alpha)
#plt.axhline(0,ls='-',color='k',lw=0.8)
#
#plt.scatter(x, epath_U0[0,:], s=scale*np.abs(evecb_U0[15,0,:])**2, c='y', alpha=alpha, label='As2-$p_{z}$')
#for i in range(1,no):
#    plt.scatter(x, epath_U0[i,:], s=scale*np.abs(evecb_U0[15,i,:])**2, c='y', alpha=alpha)
#plt.axhline(0,ls='-',color='k',lw=0.8)
#
##plt.ylabel('$E$ (eV)', size=15)
#plt.xlim(0,Nkpath)
#plt.ylim(-4.6,3.1)
##plt.xticks([0,scale,100,171,230,280,330,401],[r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$'], size=15)
##plt.yticks([-2,0,4,6],["","","",""],size=15)
#lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=12)
#for handle in lgnd.legendHandles:
#    handle.set_sizes([20.0])
#    handle.set_alpha(1.0)


plt.tight_layout()
plt.savefig('WannTB_char.pdf')
plt.show()

