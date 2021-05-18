#Maneuver V-n Diagram
#Gust V-n Diagram 

import matplotlib.pyplot as plt
import numpy as np

reg = int(input("Masukkan aturan regulasi yang anda pilih , 1 untuk FAR 25 dan 2 untuk FAR 23 : "))
#Input
MTOW = 27365.6 # Maximum Takeoff Weight (kg)
MTOW_pon = MTOW*2.205 #MTOW (pound)
V_s1_ms = 56.893 # Stall Velocity (m/s)
V_s1 = V_s1_ms*1.298 # Stall Velocity (KEAS)
V_C_ms = 149.189 # Cruise Velocity (m/s)
V_C = V_C_ms*1.298 # Cruise Velocity (KEAS)
h = 0  #aircraft cruise altitude (ft)
m= MTOW #pound
p0 = 1.225 #air density
c = 2.43  #MAC (m)
a = 4.3    #Cl gradient
S =  67.527  #lift surface area
g = 9.81 # gravity (m/s^2)

#Maximum Load Factor
n_min = -1
n_max = 2.1 + 24000/(MTOW_pon + 10000)
if (n_max > 3.8):
    n_max = 3.8

else:
    n_max = n_max

    print("Nilai maksimum load factor adalah ", n_max)

#Maneuvering Speed (Va)
V_A = V_s1 * n_max**0.5
print("Nilai Va adalah ", V_A)
print("Nilai Vc adalah ", V_C)
#setting x coordinate (V)
Va1= np.arange(0,V_s1,0.1)
#setting y coordinate (n)
y_A1= (Va1/V_s1)**2
#setting x coordinate (V)
Va2= np.arange(V_s1,V_A,0.1)
#setting y coordinate (n)
y_A2 = (Va2/V_s1)**2


#plt.show()

#Design Dive Speed (Vd)
V_D = 1.25*V_C
print("Nilai Vd adalah ", V_D)

#Negative load factor distribution

#setting x coordinate (V)
Vs= np.arange(0,V_s1,0.1)
#setting y coordinate (n)
y_s= -(Vs/V_s1)**2


#Plot maneuver V-n
plot3 = plt.figure(3)
#plt.plot(Va,y_A,'r')
plt.plot(Va1,y_A1,'r')
plt.plot(Va2,y_A2,'r')
plt.plot(Vs,y_s,'r')
plt.plot([V_A,V_D],[n_max,n_max],'b')
plt.plot([V_D,V_D],[0,n_max],'b')
plt.plot([V_D,V_C],[0,n_min],'b')
plt.plot([V_C,V_s1],[n_min,n_min],'b')
plt.plot([0,V_D],[0,0],'--b')
plt.plot([V_s1,V_s1],[n_min,0],'--y')
plt.plot([V_A,V_A],[0,n_max],'--g')
plt.plot([V_C,V_C],[n_min,n_max],'--r')

plt.xlabel('Velocity (KEAS)')
plt.ylabel('n')
plt.title('Flight manuever')
plt.show()

#Gust Vn diagram
if(reg== 1):
    if (h>15000):

        v_gc_ft =  44 + (26-44)*(h-15000)/(50000-15000)
        v_gc =  0.3048* v_gc_ft # m/s
        v_gd = 0.5* v_gc # m/s

    else :
        
        v_gc_ft =  56 + (44-56)*(h-0)/(15000-0)
        v_gc =  0.3048* v_gc_ft # m/s
        v_gd = 0.5* v_gc # m/s



else: #FAR 23
    if (h>20000):

        v_gc_ft = 50+ (25-50)*(h-20000)/(50000-20000)
        v_gc =  0.3048* v_gc_ft # m/s
        v_gd_ft = 25+ (12.5-25)*(h-20000)/(50000-20000)
        v_gd = 0.3048* v_gd_ft # m/s

    else:
        v_gc_ft = 50
        v_gc =  0.3048* v_gc_ft # m/s
        v_gd_ft = 25
        v_gd = 0.3048* v_gd_ft # m/s

#Gust equation n by V
niu = 2*m/(p0*c*a*S)
k_g = 0.88*niu/(5.3+niu)
#setting x coordinate 
Vgc= np.arange(0,V_C,0.1)
Vgd= np.arange(0,V_D,0.1)
n_gustc1 = 1+k_g*v_gc*Vgc*a*p0*S/(2*m*g)
n_gustc2 = 1-k_g*v_gc*Vgc*a*p0*S/(2*m*g)
n_gustd1 = 1+k_g*v_gd*Vgd*a*p0*S/(2*m*g)
n_gustd2 = 1-k_g*v_gd*Vgd*a*p0*S/(2*m*g)

n_gustc1_max = 1+k_g*v_gc*V_C*a*p0*S/(2*m*g)
#print ('Pada ketinggian ', h , ' Nilai n maksimum saat VC = ',n_gustc1_max)
n_gustc2_min = 1-k_g*v_gc*V_C*a*p0*S/(2*m*g)
#print ('Pada ketinggian ', h , ' Nilai n minimum saat VC = ',n_gustc2_min)
n_gustd1_max = 1+k_g*v_gd*V_D*a*p0*S/(2*m*g)
#print ('Pada ketinggian ', h , ' Nilai n maksimum saat VD = ',n_gustd1_max)
n_gustd2_min = 1-k_g*v_gd*V_D*a*p0*S/(2*m*g) 
#print ('Pada ketinggian ', h , ' Nilai n minimum saat VD = ',n_gustd2_min)


#combining
plot5=plt.figure(5)
plt.plot(Va1,y_A1,'--r')
plt.plot(Va2,y_A2,'r')
plt.plot(Vs,y_s,'--r')
plt.plot([V_A,V_D],[n_max,n_max],'b')
plt.plot([V_D,V_D],[0,n_max],'b')
plt.plot([V_D,V_C],[0,n_min],'b')
plt.plot([V_C,V_s1],[n_min,n_min],'b')
plt.plot([0,V_D],[0,0],'--b')
plt.plot([V_s1,V_s1],[n_min,1],'-.y')
plt.plot([V_A,V_A],[0,n_max],'-.g')
plt.plot([V_C,V_C],[n_gustc2_min,n_gustc1_max],'-.r')
plt.plot([V_D,V_D],[n_gustd2_min,n_gustd1_max],'b')
plt.plot(Vgc,n_gustc1,'--g')
plt.plot(Vgc,n_gustc2,'--g')
plt.plot(Vgd,n_gustd1,'--g')
plt.plot(Vgd,n_gustd2,'--g')
plt.plot([V_C,V_D],[n_gustc1_max,n_gustd1_max],'--g')
plt.plot([V_C,V_D],[n_gustc2_min,n_gustd2_min],'--g')

plt.xlabel('Velocity (KEAS)')
plt.ylabel('n')
plt.title('Flight Envelope')
plt.show()

list1=[n_gustc2_min,-1]
list2=[n_gustd2_min,0]
list3=[n_gustd1_max,n_max]
list4=[n_gustc1_max,n_max]
print("critical points (n,V) on flight envelope are : \n( 0,0 ) \n(-1,",V_s1,")\n(",min(list1),",",V_C,")\n(",min(list2),",",V_D,")\n(",max(list3),",",V_D,")\n(",max(list4),",",V_C,")\n(",n_max,",",V_A,")\n( 1,0 )")