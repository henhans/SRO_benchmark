import numpy as np
from numpy.linalg import eigh, inv
from numba import jit

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
