#Composite Diagram
#Menghasilkan gambar composite diagram

import matplotlib.pyplot as plt 
import numpy as np 
import warnings
from scipy.optimize import curve_fit
from numpy import arange
from matplotlib import pyplot
import random as random

#INPUT LINE
ct = 1.45 # tipchord wing (m)
cr = 2.80 #rootchord wing (m)
frospar = 0.25 # chord position of frontspar
respar = 0.75 # chord position of respar
elas_axis = 0.35 # chord position of elastic axis
b = 28.82 #wing span (m)
S = 67.527 #wing surface area (m^2)
Mw = 2579.2 # Wing structure mass (kg)
Ww = Mw*9.81 # Wing structure weight (N)
Pm = 3400.8 # Propulsion Mass (kg) 
Pw = Pm * 9.81 #Propulsion weight (N)
P_x = -1.421 #distance CG trust to x (LEMAC wing)
P_y = 0  #distance CG trust to y
P_z = 1.47 #distance CG trust to z

##########
h= int(input('Masukkan elemen partisi : '))

#########################################
k1 = int(input('Masukkan banyak parameter gaya angkat : ')) 

L = [0 for r in range (0,k1)]
Shear = [ 0 for r in range (0,k1)]
Bending = [ 0 for r in range (0,k1)]
Torsi = [ 0 for r in range (0,k1)]

for r in range (0,k1):
    print ("Parameter lift ke-",r+1)
    L[r] = float(input( 'Masukkan nilai lift (N) : '))
    print("##########################")
    

##########################################
k2 = int(input('Masukkan banyak penampang yang dipilih : '))

Posisi = [ 0 for i in range (0,k2)]
for i in range (0,k2):
    print ("Parameter jarak penampang ke-",i+1)
    Posisi[i] = float(input(" Masukkan jarak penampang (m) : "))

##########################################
def compositediagram(ypos):

    def liftperarea (lift):
        qCl = lift/S

        return qCl


    x_in = np.linspace (0.0,b/2,100)
    #f1 = disistribusi eliptik
    f1 = lambda x : 4*S/(3.14*b)*(1-4*(x)**2/b**2)**0.5
    #f2=distribusi planform
    f2= lambda x : ct +((cr-ct)*(0.5*b-x))/(0.5*b)
    #f3 = distribusi schfrenk
    f3 =  lambda x : 0.5*(f1(x)+f2(x))
    #f4 = c_schrenk/c_planform
    f4 = lambda x : f3(x)/f2(x)

    #weight distribution (0 m from tip chord)
    ls = lambda x : ct*x + (cr-ct)*x/b #section surface wing area
    w = lambda x : ls(x)/(S/2)*Ww 

    for r in range (0,k1):
        if (L[r]>0):
            eng = Pw
        else:
            eng = -Pw
    #menghitung elemen ke i, ada h partisi
        a=[0 for i in range (h+1)]
        for i in range (0,h+1):
            a[i] = b/(2*h)*i

        an=[0 for i in range (h+1)]
        for i in range (0,h+1):
            an[i] = -a[i]

        #weight and lift distribution    
        cl=[0 for j in range (h-1,-1,-1)]   
        for j in range (h-1,-1,-1):
            #koefisien gaya masing masing lift
            cl[j] = (f4(a[j]) + f4(a[j+1]))/2
            
        cpar=[0 for j in range (h-1,-1,-1)]
        for j in range (h-1,-1,-1):
            #rata rata chord partisi setiap elemen yang dibagi 10
            cpar[j] =(f2(a[j]) + f2(a[j+1]))/2
            
        S_el=[0 for j in range (h-1,-1,-1)]
        for j in range (h-1,-1,-1):
            #S_el adalah luas area tiap elemen partisi
            S_el[j] = b/(2*h) * cpar[j]

        L_eln=[0 for j in range (h,-1,-1)]
        w_disn=[0 for j in range (h,-1,-1)]
        L_el=[0 for j in range (h,-1,-1)]
        w_dis=[0 for j in range (h,-1,-1)]


        for j in range (h,-1,-1):
            if (j==h):
                L_el[j]=0
            else:
                #L_el adalah lift per elemen dengan m engalikan luas area per elemen dengan qCL
                L_el[j] = cl[j] * S_el[j] * liftperarea(L[r])
                w_dis[j] = S_el[j]/(S/2)*Ww

        for j in range (h,-1,-1):
            #L_el adalah lift per elemen dengan mengalikan luas area per elemen dengan qCL
            L_eln[j] = L_el[j]
            w_disn[j] = w_dis[j]

        #shear sebelah kanan lift
        S_p = [0 for j in range (h,-1,-1)] #Shear sebelum dikali Safety Faktor=1,5
        S_pf = [0 for j in range (h,-1,-1)] #Shear setelah dikali Safety Faktor=1,5
        for j in range (h,-1,-1):
            if (j == h):
                S_p[j]=0
            else:
                S_p[j] = S_p [j+1] + L_el[j]
                S_pf[j] = 1.5*S_p[j]

        #shear sebelah kanan weight
        S_w = [0 for j in range (h,-1,-1)] #Shear sebelum dikali Safety Faktor=1,5
        S_wf = [0 for j in range (h,-1,-1)] #Shear setelah dikali Safety Faktor=1,5
        for j in range (h,-1,-1):
            if (j == h):
                S_w[j]=0
            else:
                S_w[j] = S_w [j+1] + w_dis[j]
                S_wf[j] = 1.5*S_w[j]

        #plotting lift and weight based on load factor 
        yv=[0 for j in range (h,-1,-1)]
        yweight=[0 for j in range (h,-1,-1)]

        for j in range (h,-1,-1):
            if (L[r]>0):
                yv[j] = S_p[j]
                yweight[j] = -S_w[j]
            else:
                yv[j] = S_p[j]
                yweight[j] = S_w[j]

    #curve fitting method
        #LIFT
        def objective_lift(x, p1, p2, p3,p4,p5):
            return (p1 * x) + (p2 * x**2) + (p3 * x**3) + (p4 * x**4) + p5
        pol_x = a
        pol_y = yv
        #curve fit
        pol, _ = curve_fit(objective_lift,pol_x,pol_y)
        #summarize the parameter values
        p1,p2,p3,p4,p5 = pol
        

        #WEIGHT
        def objective_weight(x, q1, q2, q3):
            return (q1 * x) + ( q2 * x**2) + q3
        pol_x = a
        pol_y = yweight
        #curve fit
        pol, _ = curve_fit(objective_weight,pol_x,pol_y)
        #summarize the parameter values
        q1,q2,q3 = pol
        
        

        print (100*'#')
        ## ARAH SPANWISE
        #Luas Area dibawah kurva lift metode curvefitting
        def V_l(x):
            return (p1 * x) + (p2 * x**2) + (p3 * x**3) + (p4 * x**4) + p5
        def trapezoid_lift (x0,xn,n): #metode trapesium untuk lift
            #step size
            step = (xn-x0)/n
            #sum
            integral = V_l(x0)+V_l(xn)
            for i in range (1,n):
                k = x0 + i*step
                integral = integral + 2*V_l(k)

            #final integration value
            integral = integral * step/2
            return integral
        moment_lift = trapezoid_lift (0,b,h)
        #print (moment_lift)

        #Luas Area dibawah kurva weight metode curvefitting
        def V_w(x):
            return (q1 * x) + ( q2 * x**2) + q3
        def trapezoid_weight (x0,xn,n): #metode trapesium untuk weight
            #step size
            step = (xn-x0)/n
            #sum
            integral = V_w(x0)+V_w(xn)
            for i in range (1,n):
                k = x0 + i*step
                integral = integral + 2*V_w(k)

            #final integration value
            integral = integral * step/2
            return integral
        moment_weight = trapezoid_weight (0,b,h)

        #bending moment by lift
        B_lift = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
        for i in range (h,-1,-1):
            B_lift[i] = -trapezoid_lift (a[i],b/2,h)

        #bending moment by weight
        B_weight = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
        for i in range (h,-1,-1):
            B_weight[i] = trapezoid_weight (a[i],b/2,h)


        #Shear and bending moment by all force on y-direction
        def shear(lift,ypos):#y in range [0,b/2]
        
            if (ypos>=P_z):
                Shear = -V_l(ypos) + V_w(ypos) 
                
            else :
                Shear = -V_l(ypos) + V_w(ypos) + eng
                
            return Shear

        def bending(lift,ypos):#y in range [0,b/2]
        
            if (ypos>=P_z):
                Bending = trapezoid_lift(ypos,b/2,h) - trapezoid_weight(ypos,b/2,h)
            else :
                Bending = trapezoid_lift(ypos,b/2,h) - trapezoid_weight(ypos,b/2,h) - eng*P_z

            return Bending


        #TORSION ANALYSIS
        #asumsikan nodal torsion diperoleh dari hasil kali shear nodal ke elastic axis

        def torsi (lift,ypos):#menghitung luas area dibawah grafik shear pada suatu shearchord untuk torsi setiap section
            chord_len = cr - 2*(cr-ct)/b*ypos
            x_3D = chord_len
            y_3D = ypos
            t_len_lift= 2*V_l(y_3D)/chord_len
            
            #center of shear curve along a chordwise
            L1 = 0.25 * V_l(y_3D) # luas area dibawah kurva sebelah kiri aerodynamic center
            L2 = 0.75 * V_l(y_3D) # luas area dibawah kurva sebelah kanan aerodynamic center
            ##############
            x_cg_all = 0.5 * chord_len * ((1/3*L1+L2)/(L1+L2)) #CG dari segitiga shear sepanjang chord
            ##############

            #Tinjau lift dulu
            if (x_3D <= 0.25*chord_len):
                L1_loc = 1/2 * x_3D * 4 * x_3D/ chord_len * t_len_lift # area dibawah kurva di kiri x = x_3D
                L2_loc = V_l (y_3D)- L1_loc # area dibawah kurva di kanan x= x_3D
                x_cg_L1 = 2/3* x_3D # CG dari L1_loc
                x_cg_L2 = ((L1_loc+L2_loc)*x_cg_all - L1_loc*x_cg_L1)/L2_loc # CG dari L2_loc
                torsion_lift = -L1_loc * (elas_axis*chord_len-x_cg_L1) #- L2_loc * (x_cg_L2-elas_axis*chord_len)  #nilai torsi dari lift

            elif (x_3D >= elas_axis*chord_len):
                t_len_elas_axis = 4/3*(chord_len-elas_axis*chord_len)/chord_len*t_len_lift
                L1_loc = 1/2*0.25*chord_len*t_len_lift + (t_len_lift + t_len_elas_axis)*(elas_axis-0.25)*chord_len/2 # area dibawah di kiri x = EA
                L2_loc = (t_len_elas_axis + 4/3*(chord_len-x_3D)/chord_len*t_len_lift)*(x_3D-0.25*chord_len)/2 # area dibawah kurva di kanan x= EA
                L_from_EA_TE = 1/2*(1-elas_axis)*chord_len*t_len_elas_axis # Luas kurva dari elastic point ke trailing edge
                x_cg_L1 = (V_l(y_3D)*x_cg_all-L_from_EA_TE*(chord_len-2/3*(chord_len-elas_axis*chord_len)))/(V_l(y_3D)-L_from_EA_TE) # CG dari L2_loc
                x_cg_L2 = (((elas_axis-0.25)*chord_len*t_len_elas_axis)*(0.5*(elas_axis+0.25)*chord_len)+ (1/2*(elas_axis-0.25)*chord_len*(t_len_lift-t_len_elas_axis))*(0.5/3+elas_axis/3)*chord_len)/L2_loc # CG dari L1_loc
                torsion_lift = -(L1_loc*(elas_axis*chord_len-x_cg_L1) - L2_loc * (x_cg_L2-elas_axis*chord_len))


            else : #(0.25*chord_len < x_3D < elas_axis*chord_len)
                L2_loc = 1/2 * (chord_len-x_3D)* 4/3 * (chord_len-x_3D) / chord_len*t_len_lift # area dibawah kurva di kanan x = x_3D
                L1_loc = V_l (y_3D)- L2_loc # area dibawah kurva di kiri x= x_3D
                x_cg_L2 = chord_len - 2/3*(chord_len-x_3D) # CG dari L2_loc
                x_cg_L1 = ((L1_loc+L2_loc)*x_cg_all - L2_loc*x_cg_L2)/L1_loc # CG dari L1_loc
                torsion_lift = -L1_loc * (elas_axis*chord_len-x_cg_L1) #- L2_loc * (x_cg_L2-elas_axis*chord_len)  #nilai torsi dari lift 

            #tinjau weight
            z_weight = Ww/ ((respar-frospar)*chord_len)
            #bila x=x_3D disebelah kiri EA
            if (x_3D <= frospar*chord_len):
                torsion_weight = 0
            
            elif ((x_3D > frospar*chord_len) and (x_3D <= elas_axis*chord_len)):
                w_loc = (x_3D-frospar*chord_len)*z_weight
                x_cg_w = 0.5*(x_3D+frospar*chord_len)
                torsion_weight = w_loc * (elas_axis*chord_len-x_cg_w)

            elif ((x_3D < respar*chord_len) and (x_3D > elas_axis*chord_len)):
                w_loc_1 = (elas_axis-frospar)*chord_len*z_weight  # distribusi berat dari frontspar ke elastic point
                w_loc_2 = (x_3D-elas_axis*chord_len)*z_weight # distribusi berat dari elastic point ke x
                x_cg_w1 = 0.5*(elas_axis*chord_len+frospar*chord_len)
                x_cg_w2 = 0.5*(x_3D+elas_axis*chord_len)
                torsion_weight = w_loc_1 * (elas_axis*chord_len-x_cg_w1) - w_loc_2 * (x_cg_w2-elas_axis*chord_len)

            else :
                w_loc_1 = (elas_axis-frospar)*chord_len*z_weight  # distribusi berat dari frontspar ke elastic point
                w_loc_2 = (respar-elas_axis)*chord_len*z_weight # distribusi berat dari elastic point ke rearspar
                x_cg_w1 = 0.5*(elas_axis*chord_len+frospar*chord_len)
                x_cg_w2 = 0.5*(respar*chord_len+elas_axis*chord_len)
                torsion_weight = w_loc_1 * (elas_axis*chord_len-x_cg_w1) - w_loc_2 * (x_cg_w2-elas_axis*chord_len)

            #tinjau engine
            chord_len_eng = cr - 2*(cr-ct)/b*P_z 
            torsion_eng = eng * (-P_x + (elas_axis*chord_len_eng-0.25*chord_len_eng))
        ######################################################################
            #TORSI TOTAL
            if (y_3D <= P_z ):
                torsion_total = torsion_lift + torsion_weight
            else :
                torsion_total = torsion_lift + torsion_weight + torsion_eng

            return torsion_total

        Shear[r] = shear(L[r],ypos)
        Bending[r] = bending(L[r],ypos)
        Torsi[r] = torsi(L[r],ypos)
        
        print("Nilai Shear : ",Shear[r], " N")
        print("Nilai Bending Moment : ",Bending[r], " Nm")
        print("Nilai Torsi : ",Torsi[r], " Nm")
        print(100*'*')

    Label = [0 for i in range (k1)]
    for i in range (k1):
        Label [i] = "Lift = "+ str (L[i])
    #Membuat Grafik Composite
    #SHEAR GRAPH
    plt.figure()

    plt.subplot(131)
    x_coordinates = Shear[0:k1]
    y_coordinates = Bending[0:k1]
    for i in range (k1):
        rgb = (random.random(), random.random(), random.random())
        plt.scatter(Shear[i], Bending[i], c=[rgb], label= "Lift = "+ str (L[i])+ ' N')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Shear (N)')
    plt.ylabel('Bending Moment (Nm)')
    plt.title("YW280 Wing Load Envelope \n M-V Diagram \n y = "  + str (ypos)+ " m")
    plt.legend(frameon= True, loc='best', ncol=1)


    plt.subplot(132)
    x_coordinates = Shear[0:k1]
    y_coordinates = Torsi[0:k1]
    for i in range (k1):
        rgb = (random.random(), random.random(), random.random())
        plt.scatter(Shear[i], Torsi[i], c=[rgb], label= "Lift = "+ str (L[i])+ " N")
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Shear (N)')
    plt.ylabel('Torsion (Nm)')
    plt.title("YW280 Wing Load Envelope \n T-V Diagram \n y = " + str (ypos) + " m")
    plt.legend(frameon= True, loc='best', ncol=1)

    plt.subplot(133)
    x_coordinates = Torsi[0:k1]
    y_coordinates = Bending[0:k1]
    for i in range (k1):
        rgb = (random.random(), random.random(), random.random())
        plt.scatter(Torsi[i], Bending[i], c=[rgb], label= "Lift = "+ str (L[i])+ ' N')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Torsion (Nm)')
    plt.ylabel('Bending Moment (Nm)')
    plt.title("YW280 Wing Load Envelope \n M-T Diagram \n y = "  + str (ypos)+ " m")
    plt.legend(frameon= True, loc='best', ncol=1)
    plt.show()


#OUTPUT
for i in range (0,k2):
    print (compositediagram (Posisi [i]))
    
