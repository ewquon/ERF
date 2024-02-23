#!/usr/bin/env python
#
# Deardorff SGS model: analytical TKE solution for constant shear in a column
# of air
#
import numpy as np
import matplotlib.pyplot as plt

zhi = 1000.
Nz = 128
dz = zhi / Nz
#U0 = 5.0 # not used
dUdz = 10.0/zhi # (m/s)/m
Ck = 0.1
Ce = 0.93

# analytical result
e_exact = Ck/Ce*(dUdz*dz)**2

# verify
dt = 0.05
e = 0.1 # initial guess
tol = 1e-12
dedt = np.nan
earr = []
for i in range(999999):
    earr.append(e)
    print(i,i*dt,dedt,earr[-1])
    Km = Ck * dz * np.sqrt(e)  # eddy difusivity of momentum, Km == nu_t
    upwp = -Km * dUdz          # <u'w'> = tau_13 = -2*nu_t*S_13
    diss = Ce * e**1.5 / dz
    dedt = -upwp*dUdz - diss
    e+= dedt * dt
    if np.abs(dedt) < tol:
        break
print(f"step {i} : Km={Km} u'w'={upwp} P={-upwp*dUdz} ε={diss} de/dt={dedt} e={e}")
print('expected e =',e_exact)
print('final error:',e - e_exact)

fig,axs = plt.subplots(nrows=2,sharex=True)
times = dt*np.arange(len(earr))
axs[0].plot(times,earr)
axs[1].semilogy(times, np.abs(np.array(earr)-e_exact))
axs[0].set_ylabel('SFS e [m$^2$ s$^{-2}$]')
axs[1].set_ylabel('|error| [m$^2$ s$^{-2}$]')
axs[1].set_xlabel('time [s]')
plt.savefig('evolve_tke_SP.png',bbox_inches='tight')

# expected output:
#
# step 103172 : Km=0.010007110589932316 u'w'=-0.00020014221179864633 P=2.001422117986463e-06 ε=2.0014231179737236e-06 de/dt=-9.999872604403188e-13 e=0.0006562923305468312
# expected e = 0.000656292002688172
# final error: 3.278586591668256e-10
