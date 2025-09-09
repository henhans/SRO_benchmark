import matplotlib.pyplot as plt
import numpy as np

data_15b = np.loadtxt("gRISB/3o15b/JoverU0.2_scanU/U_occ1_occ2_occ3_Z1_Z2_Z2_ekin_epot_etot_mu_3g.dat").T
data_9b = np.loadtxt("gRISB/3o9b/JoverU0.2_scanU/U_occ1_occ2_occ3_Z1_Z2_Z2_ekin_epot_etot_mu_3g.dat").T
data_3b = np.loadtxt("gRISB/3o3b/JoverU0.2/U_occ1_occ2_occ3_Z1_Z2_Z2_ekin_epot_etot_mu_3g.dat").T

U_qmc = [0.0, 0.5,1.0,1.5,2.0,2.5,3.0]
Zxy_qmc = [1.0, 0.89, 0.63, 0.36, 0.21, 0.14, 0.10]
Zxz_qmc = [1.0, 0.80, 0.55, 0.44, 0.30, 0.21, 0.15]
Ekin_qmc = np.array([-1.251160,-1.205045,-1.131235,-1.01483,-0.901525,-0.809565,-0.734395])
#Epot_qmc = [-1.398325552001718, 1.9448-1.03*4.0-0.704001744, 3.7171-2.06*4-0.70085139, 5.4722-3.17*4-0.7029, 7.1548-4.32*4-0.70361, 8.8093-5.5*4-0.7027, 10.5075-6.7*4-0.706]
Epot_qmc = np.array([-1.398325552001718, 1.9448-2*0.704001744, 3.7171-2*0.70085139, 5.4722-2*0.7029, 7.1548-2*0.70361, 8.8093-2*0.7027, 10.5075-2*0.706])

plt.figure(figsize=(8,10))

plt.subplot(4,1,1)
plt.plot(data_3b[0], data_3b[4], '--', color='grey', label='RISB ($N_b=3$)')
plt.plot(data_9b[0], data_9b[4], 'b-', label='g-RISB ($N_b=9$)')
plt.plot(data_15b[0], data_15b[4], 'g-', label='g-RISB ($N_b=15$)')
plt.plot(U_qmc, Zxy_qmc, 'r-o', label='DMFT-CTQMC')
plt.xlim(0,3.0)
plt.ylim(0,1.01)
plt.xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0],["","","","","","",""],size=15)
plt.yticks(size=15)
#plt.xlabel(r'$U$',size=15)
plt.ylabel(r'$Z_{xy}$',size=15)
plt.legend(loc='lower left',fontsize=13)
plt.text(0.1,0.8,'(a)', size=15)

plt.subplot(4,1,2)
plt.plot(data_3b[0], data_3b[6], '--', color='grey', label='RISB')
plt.plot(data_9b[0], data_9b[6], 'b-', label='g-RISB ($N_b=9$)')
plt.plot(data_15b[0], data_15b[6], 'g-', label='g-RISB ($N_b=15$)')
plt.plot(U_qmc, Zxz_qmc, 'r-o', label='DMFT-CTQMC')
plt.xlim(0,3.0)
plt.ylim(0,1.01)
plt.xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0],["","","","","","",""],size=15)
plt.yticks(size=15)
#plt.xlabel(r'$U$',size=15)
plt.ylabel(r'$Z_{xz/yz}$',size=15)
#plt.legend(loc='lower left',fontsize=15)
plt.text(0.1,0.8,'(b)', size=15)

plt.subplot(4,1,3)
plt.plot(data_3b[0], data_3b[7], '--', color='grey', label='RISB')
plt.plot(data_9b[0], data_9b[7], 'b-', label='g-RISB ($N_b=9$)')
plt.plot(data_15b[0], data_15b[7], 'g-', label='g-RISB ($N_b=15$)')
plt.plot(U_qmc, Ekin_qmc, 'ro', label='DMFT-CTQMC')
plt.xlim(0,3.0)
#plt.ylim(0,1.01)
plt.xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0],["","","","","","",""],size=15)
plt.yticks(size=15)
#plt.xlabel(r'$U$',size=15)
plt.ylabel(r'$E_{kin}$ (eV)',size=15)
#plt.legend(loc='lower left',fontsize=15)
plt.text(0.1,-0.7,'(c)', size=15)

plt.subplot(4,1,4)
plt.plot(data_3b[0], data_3b[8], '--', color='grey', label='RISB')
plt.plot(data_9b[0], data_9b[8], 'b-', label='g-RISB ($N_b=9$)')
plt.plot(data_15b[0], data_15b[8], 'g-', label='g-RISB ($N_b=15$)')
plt.plot(U_qmc, Epot_qmc, 'ro', label='DMFT-CTQMC')
plt.xlim(0,3.0)
#plt.ylim(0,1.01)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel(r'$U$ (eV)',size=15)
plt.ylabel(r'$E_{pot}$ (eV)',size=15)
#plt.legend(loc='lower left',fontsize=15)
plt.text(0.1,9,'(d)', size=15)

#plt.subplot(5,1,5)
#plt.plot(data_3b[0], data_3b[9]+data_3b[10]*4 - Ekin_qmc - Epot_qmc, '--', color='grey', label='RISB')
#plt.plot(data_9b[0], data_9b[9]+data_9b[10]*4 - Ekin_qmc - Epot_qmc, 'b-', label='g-RISB ($N_b=9$)')
#plt.plot(data_15b[0], data_15b[8]+data_15b[10]*4 - Ekin_qmc - Epot_qmc, 'g-', label='g-RISB ($N_b=15$)')
##plt.plot(U_qmc, Ekin_qmc+Epot_qmc, 'ro', label='DMFT-CTQMC')
#plt.xlim(0,3.0)
##plt.ylim(0,1.01)
#plt.xticks(size=15)
#plt.yticks(size=15)
#plt.xlabel(r'$U$',size=15)
#plt.ylabel(r'$E_{tot}-E_{tot}^{CTQMC}$',size=15)
##plt.legend(loc='lower left',fontsize=15)
##plt.text(0.1,0.8,'(b)', size=15)

plt.tight_layout()
#plt.subplots_adjust(hspace=0.0,wspace=0.,left=0.1,right=0.98,bottom=0.1,top=0.96)
plt.savefig('Z.pdf')
plt.show()
