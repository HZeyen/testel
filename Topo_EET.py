# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:27:34 2022

@author: Hermann Zeyen
Université Paris-Saclay

This program reads a topographic profile and calculates the deformation of an
elastic plate of constant rigidity using Fourier transforms.

The force necessary to deform the plate so that the actual topography is
approximated is calculated iteratively (a theoretical topography is varied until
the difference between this theoretical topography and the deformation
corresponds to the actually measured topography).

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
if "C:\\Sources_2010\\Python_programs" not in sys.path:
    sys.path.append("C:\\Sources_2010\\Python_programs")
import HZ_routines as HZ
import tkinter as tk
from tkinter.filedialog import askopenfilename
from scipy.interpolate import interp1d

# Read topography data
dir0 = r"C:/Documents/PROJECTS/Olympus_Mons"
os.chdir(dir0)
root = tk.Tk()
topo_file = askopenfilename(title="Topography file",initialdir=dir0,
                    filetypes=(("All Files","*.*"),("dat Files","*.dat")))

root.destroy()
dir0 = os.path.dirname(topo_file)
os.chdir(dir0)
print(f"topography file chosen: {topo_file}")

with open(topo_file,"r") as fo:
    x = []
    topo = []
    while True:
        try:
            nums = fo.readline().split()
            x.append(float(nums[0]))
            topo.append(float(nums[1]))
        except:
            break
# Open dialog box for control parameters
# "equilibrate" means that topography is adjusted after flexural deformation.
#    If not equilibrqted, only the effect of topographic mass is calculated
results = HZ.Dialog(["Elastic thickness (km)",\
                     "Topo-density (kg/m3)",\
                     "Asth. density (kg/m3)",\
                     "Young's modulus (GPa)",\
                     "Poisson's ratio",\
                     "g (m/s2]",
                     "Topo central point [km]",\
                     "dT_plume [K]",\
                     "Plume radius horizontal (km)",\
                     "Plume radius vertical (km)",\
                     "Plume center position [km]",\
                     "Underplating (% of topography)",\
                     "Plotting limit(km]"],
                     ["100","3200","3500","100","0.25","3.71","20","300","1000",\
                      "500","0","100","2000"],\
                     "Control parameters")

h = float(results[0])*1000
Rho_topo = float(results[1])
Rho_a = float(results[2])
E = float(results[3])*1E9
Poisson = float(results[4])
g = float(results[5])
topo_max = float(results[6])
dT = float(results[7])
r_plume_h = float(results[8])*1000.
r_plume_v = float(results[9])*1000.
plume_c = float(results[10])*1000.
under_percent = float(results[11])*0.01
xlimit_plot = float(results[12])
T_expansion = 3.5E-5
rho_fill = 0.

x_data = np.array(x)*1000
topo = np.array(topo)
nx0 = np.where(np.abs(x_data)<900)[0][0]
topo[nx0] = topo_max*1000.
# Interpolate topography every 1 km
xmin = x_data.min()
xmax = x_data.max()
dx = 1000.
x = np.arange(xmin,xmax+1,dx)
f = interp1d(x_data, topo)
data = f(x)
topo_mass = f(x)
under_thick = data*under_percent
under_thick1 = data*0.5
nx_min = np.where(x >= -xlimit_plot*1000.)[0][0]
nx_max = np.where(x <= xlimit_plot*1000.)[0][-1]

n_dat = len(x)

# np1 = np.where(x >= -xlimit_plot)[0][0]
# np2 = np.where(x > xlimit_plot)[0][0]
# xplt = x[np1:np2]

#fig, (ax_topo, ax_eet, ax_def) = plt.subplots(2,2, sharex=True, figsize=(10,8))
fig = plt.figure(tight_layout=True, figsize=(15,12))
gs = GridSpec(15, 13, figure=fig)
# Axis for structural mode
ax_topo = fig.add_subplot(gs[:7, :6])
# Axis for elastic deformation
ax_eet = fig.add_subplot(gs[:7,7:])
# Axis for resulting topography
ax_def = fig.add_subplot(gs[8:,:])

#ax_topo.plot(x_data*0.001,topo,"b*",label="topo data")
#ax_topo.plot(x*0.001,data,"k",label="interpolated")
ax_topo.plot(x*0.001,data,"b",label="topo data")
ax_topo.plot(x*0.001,-under_thick,"c",label="underplating")
ax_topo.set_xlabel("Distance [km]")
ax_topo.set_ylabel("Topography [m]")
ax_topo.set_title("Topography")

data_u = data + under_thick
Rtg=Rho_topo*g
d_Rho = Rho_a-Rho_topo
d_rho_plume = -Rho_a*T_expansion*dT
Rag=(Rho_a-rho_fill)*g
D=E*h**3/(12*(1-Poisson**2))

dk = 2*np.pi/(n_dat*dx)
k = np.arange(n_dat,dtype=float)
Ny = int(n_dat/2)
for i in range(1,Ny):
    k[n_dat-i] = -k[i]
k = k*dk
k[0]=1
lam = 2*np.pi/k[1:Ny]
k4 = D*k**4
ftopo = Rtg/(k4+Rag)
ftopo[0] = 0
iteration = 0
comp = Rag/(Rag+k4[1:Ny])
lam05 = lam[comp>=0.5][-1]*0.001

alpha = (4*D/(g*(Rho_a-rho_fill)))**0.25
Force = (data-data.min())*dx*Rho_topo*g
Force_u = (data_u-data.min())*dx*Rho_topo*g
# Force_total = np.sum(data-data.min())*dx*g*Rho_topo
if Force.max() == 0:
    x0 = np.mean(x)
else:
    x0 = np.sum(Force*x)/np.sum(Force)

x0 = plume_c

h_plume = np.zeros(n_dat)
r_plume2_v = r_plume_v**2
x_plume = x-x0
xp = 1-(x_plume/r_plume_h)**2
for i in range(n_dat):
    if x_plume[i] < r_plume_h and x_plume[i] > -r_plume_h:
        h_plume[i] = np.sqrt(r_plume2_v*xp[i])

fplume = d_rho_plume*g/(Rho_topo*g+k4)

xx = np.abs(x-x0)/alpha
# deform_analytic_simple = -Force_total*alpha**3/(8*D)*np.exp(-xx)*(np.cos(xx)+np.sin(xx))

Force[data<0] = -data[data<0]*dx*1000
Force_u[data_u<0] = -data_u[data_u<0]*dx*1000
# fac = -Force*alpha**3/(8*D)
# deform_analytic = np.zeros(n_dat)
# for i in range(n_dat):
#     xx = np.abs(x-x[i])/alpha
#     deform_analytic += fac[i]*np.exp(-xx)*(np.cos(xx)+np.sin(xx))

Fv = np.fft.fft(topo_mass)
Fu = np.fft.fft(topo_mass+under_thick)
Ft = Fv*ftopo
Ft_u = Fu*ftopo
Fp = np.fft.fft(h_plume)
FFp = Fp*fplume
deform_topo = np.real(np.fft.ifft(Ft))
deform_topo -= deform_topo[0]
deform_topo_u = np.real(np.fft.ifft(Ft_u))
deform_topo_u -= deform_topo_u[0]
deform_plume = np.real(np.fft.ifft(FFp))
deform_plume -= deform_plume[0]
deform = deform_topo_u+deform_plume
t_elast_plume = topo_mass-deform_plume
t_elast_topo = topo_mass-deform_topo
t_elast_topo_u = topo_mass-deform_topo_u+under_thick
t_elast = topo_mass-deform+under_thick
nbuldge_neg = np.argmin(deform_topo[x<x0])
xbuldge_neg = x[nbuldge_neg]*0.001
buldge_neg = -deform_topo[nbuldge_neg]
yy = deform_topo*1
yy[x<x0] = 0
nbuldge_pos = np.argmin(yy)
xbuldge_pos = x[nbuldge_pos]*0.001
buldge_pos = -deform_topo[nbuldge_pos]


ax_topo.plot(x*0.001,-h_plume*0.1,"r",label="plume (thickness/10)")

ax_topo.legend(loc="lower right")
ax_topo.set_xlim(-xlimit_plot,xlimit_plot)

ax_eet.plot(x*0.001,-deform_topo,"b",label="Topography")
ax_eet.plot(x*0.001,-deform_topo_u,"c",label="Topography+underplating")
ax_eet.plot(x*0.001,-deform_plume,"r",label="Plume")
ax_eet.plot(x*0.001,-deform,"g",label="Total")
# ax_eet.plot(x*0.001,deform_analytic,"m",label="Analytic")
# ax_eet.plot(x*0.001,deform_analytic_simple,"k",label="Analytic_simple")
ax_eet.set_xlabel("Distance [km]")
ax_eet.set_ylabel("Plate deformation [m]")
ax_eet.set_title("Elastic deformation")
ax_eet.plot([xbuldge_neg,xbuldge_neg],[ax_eet.get_ylim()[0],buldge_neg],"k--",label="buldge position")
ax_eet.plot([xbuldge_pos,xbuldge_pos],[ax_eet.get_ylim()[0],buldge_pos],"k--")
ax_eet.legend(loc="lower right")
ax_eet.set_xlim(-xlimit_plot,xlimit_plot)

ax_def.plot(x*0.001,t_elast_topo,"b",label="topo surface mass")
ax_def.plot(x*0.001,t_elast_topo_u,"c",label="topo surface+underplating")
ax_def.plot(x*0.001,t_elast_plume,"r",label="topo plume")
ax_def.plot(x*0.001,t_elast,"g",label="topo total")
ax_def.set_xlabel("Distance [km]")
ax_def.set_ylabel("Topography [m]")
ax_def.set_title(f"Resulting topgraphy (EET = {h/1000:0.1f}km, g={g:0.2f}m/s2)")
ax_def.legend(loc="upper right")
ax_def.set_xlim(-xlimit_plot,xlimit_plot)

fig.savefig(f"Elastic_deformation_EET{h*0.001:0.0f}_PRv{r_plume_v*0.001:0.0f}_PT{dT:0.0f}.png")

with open("EET_results.dat","w") as fo:
    fo.write(f"EET: {h/1000:0.1f} km; g={g:0.2f} m/s2\n")
    fo.write("       X     topo   def_topo   topo+def  underplating  def_under  "+\
             "topo+under+def      plume  def_plume  topo+under+plume+def"+\
             "  Total_deformation  Total topo\n")
    for i in range(nx_min,nx_max):
        fo.write(f"{x[i]*0.001:8.0f} {data[i]:8.2f} {-deform_topo[i]:10.2f} "+\
                 f"{t_elast_topo[i]:10.2f} {-under_thick[i]:13.2f} "+\
                 f"{-deform_topo_u[i]:10.2f} {t_elast_topo_u[i]:15.2f}"+\
                 f"{-h_plume[i]:11.2f} {-deform_plume[i]:21.2f}"+\
                 f"{t_elast_plume[i]:10.2f} {-deform[i]:18.2f} {t_elast[i]:11.2f}\n")

# fig2, ax2 = plt.subplots(1,1, figsize=(10,8))
# ax2.plot(lam*0.001,comp,label="compensation")
# ax2.plot([lam05,lam05],[0,1],"r",label=f"lambda 1/2 = {lam05:0.1f}km")
# ax2.set_xlabel("Wavelength [km]")
# ax2.set_ylabel("compensation")
# ax2.set_title(f"Elastic/local deformation (EET = {h/1000:0.1f}km, g={g:0.2f}m/s2)")
# ax2.legend(loc="lower right")

i1 = np.where(x>=-280000.)[0][0]-2
i2 = i1+5
for i in range(i1,i2):
    print(f"{x[i]*0.001:0.0f}: {t_elast_topo[i]:10.2f}, {-deform_plume[i]:0.2f} ")


