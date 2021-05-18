
#FLAPS MANEUVER
#Dibuat mengikuti kaidah FAR 25

import matplotlib.pyplot as plt
import numpy as np

#INPUT DATA
print('Masukkan kondisi flap anda \n 1. Takeoff dengan MTOW \n 2. Mendekati landing dengan MLW ')
kon = int(input('yaitu : '))
MTOW = 27365.6 # Maximum Takeoff Weight (kg)
MLW = 27165.6 #Maximum Landing Weight (kg)
g = 9.81 # gravity
pro = 1.225 # air density (kg/m^3)
Cl_max = 2.423 # Cl maximum value (/rad)
S = 67.527 # Wing Surface area (m^2)
n_max_flap = 2 # max load factor in flap maneuver

#ANALYSIS
if (kon==1):
    V_st = (2*MTOW*g/(pro*S*Cl_max))**0.5
    V_st = 1.94 * V_st # KEAS
    V_F = 1.6*V_st #KEAS
else :
    V_st = (2*MLW*g/(pro*S*Cl_max))**0.5
    V_st = 1.94 * V_st # KEAS
    V_F = 1.8*V_st #KEAS
print ('Nilai V stall adalah ', V_st, "KEAS")
print ('Nilai VF adalah ', V_F, "KEAS")

V_f1 = V_st * n_max_flap **0.5
#setting x coordinate (V)
Vf= np.arange(0,V_f1,0.1)
#setting y coordinate (n)
y_f = (Vf/V_st)**2

plot1 = plt.figure(1)
plt.plot(Vf,y_f,'g')
plt.plot([V_f1,V_F],[n_max_flap,n_max_flap],'g')
plt.plot([V_F,V_F],[n_max_flap,0],'g')
plt.xlabel('Velocity (KEAS)')
plt.ylabel('n')
if (kon ==1) :
    plt.title('Flaps manuever \n Takeoff with MTOW')
else :
    plt.title('Flaps maneuver \n Approaching for landing with MLW')
plt.show ()


