#GUST DIAGRAM
#Untuk membandingkan dua ketinggian untuk kasus gust maneuver


import matplotlib.pyplot as plt
import numpy as np

reg = int(input("Masukkan aturan regulasi yang anda pilih , 1 untuk FAR 25 dan 2 untuk FAR 23 : " ))
m= 27365.6  #MTOW (kg)
p0 = 1.225 #air density sea level
p1 = 0.549 #air density at 25000 ft
c = 2.43  #MAC (m)
a = 4.3    #Cl gradient
S =  67.527 #lift surface area
g = 9.81 # gravity (m/s^2) 
V_C_ms = 149.189  # Cruise Velocity (m/s)
V_C = V_C_ms*1.298 # Cruise Velocity (KEAS)
V_s1_ms = 56.893 # Stall Velocity (m/s)
V_s1 = V_s1_ms*1.298 # Stall Velocity (KEAS)
V_D= 1.25*V_C

plot1 = plt.figure(1)
if(reg== 1):
    
    plt.plot ([0,15000,50000],[56,44,26],'b',label = 'Cruise')
    plt.plot ([0,15000,50000],[28,22,13],'r',label= 'Dive')

else: #FAR 23
    
    plt.plot ([0,20000,50000],[50,50,25],'b',label = 'Cruise')
    plt.plot ([0,20000,50000],[25,25,12.5],'r',label = 'Dive')

plt.ylabel('Velocity (ft/s)')
plt.xlabel('h (ft)')
plt.title('Reference Gust Velocities ')
plt.legend(frameon= True, loc='best', ncol=2)
plt.show()
    
    
def gust(h):
    if (h== h1):
        p= p1
    else:
        p=p2
    niu = 2*m/(p*c*a*S)
    k_g = 0.88*niu/(5.3+niu)

    if(reg== 1):
        
        if (h>15000):
            v_gc_ft = lambda h :  44 + (26-44)*(h-15000)/(50000-15000)
            v_gd_ft = lambda h : 0.5 * v_gc_ft(h)
            v_gc = lambda h : 0.3048* v_gc_ft (h) # m/s
            v_gd = lambda h : 0.5* v_gc (h) # m/s

        else :
            v_gc_ft = lambda h : 56 + (44-56)*(h-0)/(15000-0)
            v_gd_ft = lambda h : 0.5 * v_gc_ft(h)
            v_gc =  lambda h : 0.3048* v_gc_ft (h) # m/s
            v_gd =  lambda h : 0.5* v_gc (h)  # m/s

    else: #FAR 23
        
        if (h > 20000):
            v_gc_ft = lambda h : 50+ (25-50)*(h-20000)/(50000-20000)
            v_gc =  lambda h : 0.3048* v_gc_ft (h) # m/s
            v_gd_ft = lambda h : 25+ (12.5-25)*(h-20000)/(50000-20000)
            v_gd = lambda h : 0.3048* v_gd_ft (h) # m/s

        else:
            v_gc_ft = lambda h : 50
            v_gc = lambda h : 0.3048* v_gc_ft(h) # m/s
            v_gd_ft = lambda h : 25
            v_gd = lambda h : 0.3048* v_gd_ft(h) # m/s


    Vgc= np.arange(0,V_C,0.1)
    Vgd= np.arange(0,V_D,0.1)
    n_gustc1 = 1+k_g*v_gc(h)*Vgc*a*p*S/(2*m*g)
    n_gustc2 = 1-k_g*v_gc(h)*Vgc*a*p*S/(2*m*g)
    n_gustd1 = 1+k_g*v_gd(h)*Vgd*a*p*S/(2*m*g)
    n_gustd2 = 1-k_g*v_gd(h)*Vgd*a*p*S/(2*m*g)

    n_gustc1_max = 1+k_g*v_gc(h)*V_C*a*p*S/(2*m*g)
    print ('Pada ketinggian ', h , ' Nilai n maksimum saat VC yaitu n = ',n_gustc1_max)
    n_gustc2_min = 1-k_g*v_gc(h)*V_C*a*p*S/(2*m*g)
    print ('Pada ketinggian ', h , ' Nilai n minimum saat VC yaitu n = ',n_gustc2_min)
    n_gustd1_max = 1+k_g*v_gd(h)*V_D*a*p*S/(2*m*g)
    print ('Pada ketinggian ', h , ' Nilai n maksimum saat VD yaitu n = ',n_gustd1_max)
    n_gustd2_min = 1-k_g*v_gd(h)*V_D*a*p*S/(2*m*g) 
    print ('Pada ketinggian ', h , ' Nilai n minimum saat VD yaitu n = ',n_gustd2_min)

    #Gust maximum intensity
    V_B = V_s1 * (1+k_g*v_gc(h)*V_C*a/(498*m*g))**0.5
    #print('Pada ketinggian ', h , ' Nilai intensitas maksimum gust VB = ', V_B)
    
    
    plt.plot([V_C,0,V_C],[n_gustc1_max,1,n_gustc2_min], label=" VC "+ str (h) + ' ft')
    plt.plot([V_D,0,V_D],[n_gustd1_max,1,n_gustd2_min], label = "VD " + str (h) + ' ft')  
    plt.plot([V_C,V_C],[n_gustc2,n_gustc1],color ='black', linestyle = 'dashed')
    plt.plot([V_D,V_D],[n_gustd2,n_gustd1],color ='black', linestyle = 'dashed')
    plt.xlabel('Velocity (KEAS)')
    plt.ylabel('n')
    plt.title('Gust manuever ')


        
h1 = float(input("Masukkan ketinggian uji pertama (ft) : " ))
h2 = float(input("Masukkan ketinggian uji kedua (ft) : " ))
p1 = float(input("Masukkan massa jenis ketinggian uji pertama dalam SI  : " ))
p2 = float(input("Masukkan massa jenis ketinggian uji kedua dalam SI : " ))
plot2 = plt.figure(2)
print(gust(h1))
print(gust(h2))
plt.legend(frameon= True, loc='upper left', ncol=2)
plt.show()
