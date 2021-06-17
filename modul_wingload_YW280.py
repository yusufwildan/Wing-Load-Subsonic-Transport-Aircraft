#Menghasilkan nodal  load
#Nilai shear, bending moment, dan torsi sayap

import matplotlib.pyplot as plt #supaya fungsinya seperti matlab
import numpy as np #supaya bisa define np
import warnings
from scipy.optimize import curve_fit
from numpy import arange
from matplotlib import pyplot

#INPUT LINE
ct = 1.45 # tipchord wing (m)
cr = 2.80 #rootchord wing (m)
frospar = 0.25 # chord position of frontspar
respar = 0.75 # chord position of respar
elas_axis = 0.35 # chord position of elastic axis
b = 28.82 #wing span (m)
S = 67.527 #wing surface area (m^2)
L = -291193.09 # Input lift (N)1
Mw = 2579.2 # Wing structure mass (kg)
Ww = Mw*9.81 # Wing structure weight (N)
Pm = 3400.8 # Propulsion Mass (kg) 
Pw = Pm * 9.81 #Propulsion weight (N)
P_x = -1.421 #distance CG trust to x (LEMAC wing)
P_y = 0  #distance CG trust to y
P_z = 1.47 #distance CG trust to z
qCl = L/S
##########
h= int(input('Masukkan elemen partisi : '))

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

plot1 = plt.figure(1)
plt.plot (x_in,f1(x_in),'b',label = 'Elliptic')
plt.plot (x_in,f2(x_in),'g',label = 'Planform')
plt.plot (x_in,f3(x_in),'r',label = 'Schrenk')
plt.xlabel ('Semispan (m)')
plt.ylabel ('cCla')
plt.title('Schrenk Lift Distribution')
plt.legend(frameon= True, loc='best', ncol=3)


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
    
cpar=[0 for k1 in range (h-1,-1,-1)]
for k1 in range (h-1,-1,-1):
    #rata rata chord partisi setiap elemen yang dibagi 10
    cpar[k1] =(f2(a[k1]) + f2(a[k1+1]))/2
    
S_el=[0 for k2 in range (h-1,-1,-1)]
for k2 in range (h-1,-1,-1):
    #S_el adalah luas area tiap elemen partisi
    S_el[k2] = b/(2*h) * cpar[k2]


L_el=[0 for k3 in range (h,-1,-1)]
w_dis=[0 for k3 in range (h,-1,-1)]
for k3 in range (h,-1,-1):
    if (k3==h):
        L_el[k3]=0
    else:
        #L_el adalah lift per elemen dengan m engalikan luas area per elemen dengan qCL
        L_el[k3] = cl[k3] * S_el[k3] * (L/S)
        w_dis[k3] = S_el[k3]/(S/2)*Ww
L_eln=[0 for k3 in range (h,-1,-1)]
w_disn=[0 for k3 in range (h,-1,-1)]
for k3 in range (h,-1,-1):
    #L_el adalah lift per elemen dengan mengalikan luas area per elemen dengan qCL
    L_eln[k3] = L_el[k3]
    w_disn[k3] = w_dis[k3]

#shear sebelah kanan lift
S_p = [0 for k3 in range (h,-1,-1)] #Shear sebelum dikali Safety Faktor=1,5
S_pf = [0 for k3 in range (h,-1,-1)] #Shear setelah dikali Safety Faktor=1,5
for k3 in range (h,-1,-1):
    if (k3 == h):
        S_p[k3]=0
    else:
        S_p[k3] = S_p [k3+1] + L_el[k3]
        S_pf[k3] = 1.5*S_p[k3]

#shear sebelah kanan weight
S_w = [0 for k3 in range (h,-1,-1)] #Shear sebelum dikali Safety Faktor=1,5
S_wf = [0 for k3 in range (h,-1,-1)] #Shear setelah dikali Safety Faktor=1,5
for k3 in range (h,-1,-1):
    if (k3 == h):
        S_w[k3]=0
    else:
        S_w[k3] = S_w [k3+1] + w_dis[k3]
        S_wf[k3] = 1.5*S_w[k3]

#plotting lift and weight based on load factor 
yv=[0 for i in range (h,-1,-1)]
yweight=[0 for i in range (h,-1,-1)]
for i in range (h,-1,-1):
    if (L>0):
        yv[i] = S_p[i]
        yweight[i] = -S_w[i]
    else:
        yv[i] = S_p[i]
        yweight[i] = S_w[i]


plot2 = plt.figure(2)
plt.plot(a,yv,'r')
plt.plot(an,yv,'r')
plt.xlabel('Spanwise (m)')
plt.ylabel('Shear (N)')
plt.title("Shear Lift Distribution")

print(100*'#')
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
#print (moment_weight)

#bending moment by lift
B_lift = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
for i in range (h,-1,-1):
    B_lift[i] = -trapezoid_lift (a[i],b/2,h)

#bending moment by weight
B_weight = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
for i in range (h,-1,-1):
    B_weight[i] = trapezoid_weight (a[i],b/2,h)

#engine effect
if (L>0):
    eng = Pw
else:
    eng = -Pw
#Shear and bending moment by all force on y-direction
def nodal(ypos):#y in range [0,b/2]
    
    if (ypos>=P_z):
        Shear = -V_l(ypos) + V_w(ypos) 
        Bending = trapezoid_lift(ypos,b/2,h) - trapezoid_weight(ypos,b/2,h)
    else :
        Shear = -V_l(ypos) + V_w(ypos) + eng
        Bending = trapezoid_lift(ypos,b/2,h) - trapezoid_weight(ypos,b/2,h) - eng*P_z

    print('Nilai shear force pada ', ypos ,' m dari root chord adalah ', Shear, ' N' )
    print('Nilai bending moment pada ', ypos ,' m dari root chord adalah ', Bending, ' Nm' )

#Graph of all forces
S_total = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
for i in range (h,-1,-1):
    if (a[i]>=P_z):
        S_total[i] = -V_l(a[i]) + V_w(a[i])
    else :
        S_total[i] = -V_l(a[i]) + V_w(a[i]) + eng

B_total = [0 for i in range (h,-1,-1)] #sebelum dikali Safety Faktor=1,5
for i in range (h,-1,-1):
    if (a[i]>=P_z):
        B_total[i] = trapezoid_lift(a[i],b/2,h) - trapezoid_weight(a[i],b/2,h)
    else :
        B_total[i] = trapezoid_lift(a[i],b/2,h) - trapezoid_weight(a[i],b/2,h) - eng*P_z



k1 = int(input('Masukkan berapa nodal point arah spanwise yang akan dievaluasi : '))
y_input =[0 for i in range (k1)]
for i in range (0,k1):
    print ("Nodal point ke-",i+1)
    y_input[i] = float(input("Jarak dari nodal point ke root chord (m) : "))
    print("##########################")

for i in range (0,k1):
    print(nodal(y_input[i]))

### ARAH CHORDWISE
def shearchord_lift (x_3D,y_3D):
    #chord length
    chord_len = cr - 2*(cr-ct)/b*y_3D
    t_len_lift= 2*V_l(y_3D)/chord_len

    #asumsikan LE sebagai x_3D = 0 dan y dalam arah spanwise
    if (x_3D <= frospar*chord_len):
        
        #LIFT
        z_lift = 4*x_3D/chord_len*t_len_lift #Nilai v (N/m)
        dist_shear_lift = 1/2*x_3D*z_lift

    elif ((x_3D > respar*chord_len) and (x_3D <= chord_len)):

        z_lift = 4/3*(chord_len-x_3D)/chord_len*t_len_lift #Nilai v (N/m)
        dist_shear_lift = V_l(y_3D)-1/2*(chord_len-x_3D)*z_lift
        
    elif ((x_3D > frospar*chord_len ) and (x_3D <= respar*chord_len)):
        z_lift = 4/3*(chord_len-x_3D)/chord_len*t_len_lift #Nilai v (N/m)
        dist_shear_lift = V_l(y_3D)-1/2*(chord_len-x_3D)*z_lift
    
    else :
        dist_shear_lift = 0
    return dist_shear_lift

def momenchord_lift (x_3D,y_3D):
    chord_len = cr - 2*(cr-ct)/b*y_3D
    t_len_momlift= 2*trapezoid_lift(y_3D,b/2,h)/chord_len

    #asumsikan LE sebagai x_3D = 0 dan y dalam arah spanwise
    if (x_3D <= frospar*chord_len):
        
        #LIFT
        z_momlift = 4*x_3D/chord_len*t_len_momlift #Nilai v (N/m)
        dist_shear_momlift = 1/2*x_3D*z_momlift

    elif ((x_3D > respar*chord_len) and (x_3D <= chord_len)):

        z_momlift = 4/3*(chord_len-x_3D)/chord_len*t_len_momlift #Nilai v (N/m)
        dist_shear_momlift =trapezoid_lift(y_3D,b/2,h) -1/2*(chord_len-x_3D)*z_momlift
        
    elif ((x_3D > frospar*chord_len ) and (x_3D <= respar*chord_len)):
        z_momlift = 4/3*(chord_len-x_3D)/chord_len*t_len_momlift #Nilai v (N/m)
        dist_shear_momlift = trapezoid_lift(y_3D,b/2,h)-1/2*(chord_len-x_3D)*z_momlift

    else :
        dist_shear_momlift = 0
    return dist_shear_momlift

def shearchord_weight (x_3D,y_3D):
    #chord length
    chord_len = cr - 2*(cr-ct)/b*y_3D

    #asumsikan LE sebagai x_3D = 0 dan y dalam arah spanwise
    if (x_3D <= frospar*chord_len):
        
        z_weight = 0
        dist_shear_weight = 0

    elif ( (x_3D > respar*chord_len) and (x_3D <= chord_len)):

        z_weight = V_w(y_3D)/ ((respar-frospar)*chord_len)
        dist_shear_weight = V_w(y_3D)
        
    elif ((x_3D > frospar*chord_len ) and (x_3D <= respar*chord_len)) :
        z_weight = V_w(y_3D) / ((respar-frospar)*chord_len)
        dist_shear_weight = (x_3D-frospar*chord_len)*z_weight

    else :
        dist_shear_weight = 0

    return dist_shear_weight   


def momenchord_weight (x_3D,y_3D):
    #chord length
    chord_len = cr - 2*(cr-ct)/b*y_3D

    #asumsikan LE sebagai x_3D = 0 dan y dalam arah spanwise
    if (x_3D <= frospar*chord_len):
        
        z_momweight = 0
        dist_shear_momweight = 0

    elif ( (x_3D > respar*chord_len) and (x_3D <= chord_len)):

        z_momweight = trapezoid_weight(y_3D,b/2,h) / ((respar-frospar)*chord_len)
        dist_shear_momweight = trapezoid_weight(y_3D,b/2,h)
        
    elif ((x_3D > frospar*chord_len ) and (x_3D <= respar*chord_len)) :
        z_momweight = trapezoid_weight(y_3D,b/2,h) / ((respar-frospar)*chord_len)
        dist_shear_momweight = (x_3D-frospar*chord_len)*z_momweight

    else :
        dist_shear_momweight = 0

    return dist_shear_momweight



print(100*'#')

k2 = int(input('Masukkan berapa nodal point arah chordwise yang akan dievaluasi : '))
x_3D_input =[0 for i in range (k2)]
y_3D_input =[0 for i in range (k2)]
chordlen =[0 for i in range (k2)]
front = [0 for i in range (k2)]
rear = [0 for i in range (k2)]
for i in range (0,k2):
    print ("Nodal point ke-",i+1)
    y_3D_input[i] = float(input(" ordinat titik pada sayap (m): "))
    chordlen[i] = (cr - 2*(cr-ct)/b*y_3D_input[i])
    front[i]= frospar*chordlen[i]
    rear[i]= respar*chordlen[i]
    print("Masukkan x dalam interval [0,",chordlen[i],"]")
    print("Front Spar ada pada x = ", front[i])
    print("Rear Spar ada pada x = ",rear[i])
    x_3D_input[i] = float(input(" absis titik pada sayap (m) : "))
    
    print("##########################")


#TORSION ANALYSIS
#asumsikan nodal torsion diperoleh dari hasil kali shear nodal ke elastic axis

def integ_torsi (x_3D,y_3D):#menghitung luas area dibawah grafik shear pada suatu shearchord untuk torsi setiap section
    chord_len = cr - 2*(cr-ct)/b*y_3D
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

    
######################################################################
    #TORSI TOTAL
        
    torsion_total = torsion_lift + torsion_weight


    return torsion_total

chord_len_eng = cr - 2*(cr-ct)/b*P_z 
torsion_eng = eng * (-P_x + (elas_axis*chord_len_eng-0.25*chord_len_eng))
        
sheartotalchord = [0 for i in range (0,k2)]
momentotalchord = [0 for i in range (0,k2)]
torsitotalchord = [0 for i in range (0,k2)]
C= [0 for i in range (0,k2)]
for i in range (0,k2):  
    C[i]= x_3D_input[i]/(cr - 2*(cr-ct)/b*y_3D_input[i])
    if (y_3D_input[i] >= P_z):
        sheartotalchord[i] = -shearchord_lift(x_3D_input[i],y_3D_input[i]) + shearchord_weight(x_3D_input[i],y_3D_input[i])
        momentotalchord[i] = momenchord_lift (x_3D_input[i],y_3D_input[i]) - momenchord_weight (x_3D_input[i],y_3D_input[i])
        torsitotalchord[i] = integ_torsi(x_3D_input[i],b/2) - integ_torsi(x_3D_input[i],y_3D_input[i])

    else :
        sheartotalchord[i] = -shearchord_lift(x_3D_input[i],y_3D_input[i]) + shearchord_weight(x_3D_input[i],y_3D_input[i]) + eng 
        momentotalchord[i] = momenchord_lift (x_3D_input[i],y_3D_input[i]) - momenchord_weight (x_3D_input[i],y_3D_input[i]) - eng*P_z
        torsitotalchord[i] = integ_torsi(x_3D_input[i],b/2) - integ_torsi(x_3D_input[i],y_3D_input[i]) + torsion_eng


    print ('Nodal load ke -', i+1)
    print ('x berada pada ', C[i], 'c dan y pada jarak ', y_3D_input[i],' dari fuselage' )
    print('Nodal shear pada ( ',x_3D_input[i],' , ',y_3D_input[i],' ) adalah ', sheartotalchord[i], ' N')
    print ('Nodal momen pada ( ',x_3D_input[i],' , ',y_3D_input[i],' ) adalah ', momentotalchord[i], ' Nm' )  
    print ('Nodal torsi pada ( ',x_3D_input[i],' , ',y_3D_input[i],' ) adalah ', torsitotalchord[i], ' Nm' )

    print ('###########################################################')

print(100*'#')
chord_leng = [0 for i in range (h+1)]
T = [0 for i in range (h+1)]
for i in range (0,h+1) :
    chord_leng[i] = cr - 2*(cr-ct)/b*a[i] #chord length per section
    chordlengtip = cr - 2*(cr-ct)/b*b/2
    if (a[i] >= P_z): 
        T[i] = (integ_torsi(chordlengtip,b/2) - integ_torsi(chord_leng[i],a[i]))
    else :
        T[i] = (integ_torsi(chordlengtip,b/2) - integ_torsi(chord_leng[i],a[i])) + torsion_eng
     # minus itu vektor penampang

print ('###########################################################')

chord_leng = [0 for i in range (k1)]
Torsi = [0 for i in range (k1)]
for i in range (0,k1):
    chord_leng[i] = cr - 2*(cr-ct)/b*y_input[i] #chord length per section
    chordlengtip = cr - 2*(cr-ct)/b*b/2
    if (y_input[i] >= P_z): 
        Torsi[i] = (integ_torsi(chordlengtip,b/2) - integ_torsi(chord_leng[i],y_input[i]))
    else :
        Torsi[i] = (integ_torsi(chordlengtip,b/2) - integ_torsi(chord_leng[i],y_input[i])) + torsion_eng
    print('Nilai torsi pada ', y_input[i] ,' m dari root chord adalah ', Torsi[i], ' Nm')

# TINJAU BESAR SHEAR DAN MOMENT PADA SAMBUNGAN WING DAN FUSELAGE
# Asumsi sambungan sebagai cantilever
# Vj dan Mj adalah besar shear dan moment dalam sambungan
Vj = Pw + Ww - L
Mj = -(trapezoid_lift(0,b/2,h) - trapezoid_weight(0,b/2,h) - eng*P_z)
print ('Besar shear pada sambungan wing-fuselage adalah ', Vj, ' N')
print ('Besar momen pada sambungan wing-fuselage adalah ', Mj, ' Nm')

print(100*'#')
#NINJAU WING BOX
#SHEAR
# Untuk upper dan lower berturut turut masukkan z_3D = 1 dan z_3D = 2
def shearchord_box (x_3D,y_3D,z_3D):
    #chord length
    chord_len = cr - 2*(cr-ct)/b*y_3D
    t_len_lift= 2*V_l(y_3D)/chord_len

    #asumsikan LE sebagai x_3D = 0 dan y dalam arah spanwise
    if (x_3D <= frospar*chord_len):
        
        #LIFT
        z_lift = 4*x_3D/chord_len*t_len_lift #Nilai v (N/m)
        z_weight = 0
        dist_shear_lift = 1/2*x_3D*z_lift
        dist_shear_weight = 0

    elif ( x_3D > respar*chord_len):

        z_lift = 4/3*(chord_len-x_3D)/chord_len*t_len_lift #Nilai v (N/m)
        z_weight = V_w(y_3D)/ ((respar-frospar)*chord_len)
        dist_shear_lift = V_l(y_3D)-1/2*(chord_len-x_3D)*z_lift
        dist_shear_weight = V_w(y_3D)
        
    else :
        z_lift = 4/3*(chord_len-x_3D)/chord_len*t_len_lift #Nilai v (N/m)
        z_weight = V_w(y_3D) / ((respar-frospar)*chord_len)
        dist_shear_lift = V_l(y_3D)-1/2*(chord_len-x_3D)*z_lift
        dist_shear_weight = (x_3D-frospar*chord_len)*z_weight
    

    if (z_3D == 1):
        if (y_3D > P_z):
            shear_box = -2/3 *dist_shear_lift + 1/2*dist_shear_weight 
    
        else :
            shear_box = -2/3*dist_shear_lift + 1/2*dist_shear_weight 
        
    else :
        if (y_3D > P_z):
            shear_box = -1/3 *dist_shear_lift + 1/2*dist_shear_weight 
    
        else :
            shear_box = -1/3 *dist_shear_lift + 1/2*dist_shear_weight  +eng

    return shear_box

print (100*'#')
k3 = int(input('Masukkan berapa nodal point dalam internal wing arah chordwise yang akan dievaluasi : '))
y_position = float(input('Masukkan jarak chordwise dari root chord (m) : '))
C = [0 for i in range (k3) ]
x_3D_input =[0 for i in range (k3)]
y_3D_input =[0 for i in range (k3)]
z_3D_input =[1 for i in range (k3)]
print('Masukkan titik dengan interval ', frospar,'c hingga ', respar,'c')
for i in range (0,k3):
    print ("Nodal point ke-",i+1)
    C[i] = float(input(" Masukkan persentase chord yang anda inginkan : "))
    y_3D_input[i] = y_position
    x_3D_input[i]=  C[i] * (cr - 2*(cr-ct)/b*y_3D_input[i])
    
    print("##########################")

#ALGORITMA 1 (OUTPUT DESKRIPSI TITIK)

chord_len= [0 for i in range (k3)]
Fz=[0 for i in range (k3)]
Am=[0 for i in range (k3)]
for i in range (0,k3): 
    #ALGORITMA 1 (OUTPUT DESKRIPSI TITIK) 
    chord_len[i]= (cr - 2*(cr-ct)/b*y_3D_input[i])
    camber = 0.2 #camber height of section
    Am[i] = (cr - 2*(cr-ct)/b*y_3D_input[i])*camber #enclosed area by torque
    Fz [i] = shearchord_box(x_3D_input[i],y_3D_input[i],z_3D_input[i])


    print ('Nodal load ke -', i+1)
    print ('x berada pada ', C[i], 'c dan y pada jarak ', y_3D_input[i],' dari fuselage' )
    
    print ('Nodal Fz pada ( ',x_3D_input[i],' , ',y_3D_input[i],' ) upper skin adalah ', Fz [i], ' N' ) 


    z_3D_input[i] += 1
    Fz [i] = shearchord_box(x_3D_input[i],y_3D_input[i],z_3D_input[i])
    
    print ('Nodal Fz pada ( ',x_3D_input[i],' , ',y_3D_input[i],' ) lower skin adalah ', Fz [i], ' N' )  

    print ('###########################################################')



plot5 = plt.figure(5)
plt.plot(a,S_total,'r')
plt.xlabel('Spanwise (m)')
plt.ylabel('Shear (N)')
plt.title("Shear distribution on wing")


plot6 = plt.figure(6)
plt.plot(a,B_total,'r')
plt.xlabel('Spanwise (m)')
plt.ylabel('Bending (Nm)')
plt.title("Bending distribution on wing")



plot7 = plt.figure(7)
plt.plot(a,T,'r')
plt.xlabel('Spanwise (m)')
plt.ylabel('Torsion (Nm)')
plt.title("Torsion distribution on wing")

plt.show()


