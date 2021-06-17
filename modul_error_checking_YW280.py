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
L =  766764.47# Input lift (N)1
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
        L_el[k3] = cl[k3] * S_el[k3] * -(L/S)
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
        yv[i] = -S_p[i]
        yweight[i] = -S_w[i]
    else:
        yv[i] = S_p[i]
        yweight[i] = S_w[i]




#curve fitting method
#LIFT
def objective_lift(x, p1, p2, p3):
	return (p1 * x) + (p2 * x**2) + p3
pol_x = a
pol_y = yv
#curve fit
pol, _ = curve_fit(objective_lift,pol_x,pol_y)
#summarize the parameter values
p1,p2,p3 = pol
print('y = %.5f * x + %.5f * x^2  + %.5f' % (p1, p2, p3))
# plot input vs output
x=pol_x
y=pol_y
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(0, max(x), 1)
# calculate the output for the range
y_line = objective_lift(x_line, p1, p2, p3)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red',label = 'Curve Fitting Lift Result')
pyplot.legend(frameon= True, loc='best')
pyplot.show()

#ERROR CHECKING
error_lift= [0 for i in range (h,-1,-1)]
for i in range (h,-1,-1):
    error_lift[i] = (yv [i]-objective_lift(a[i],p1,p2,p3))/yv [i]*100
    print(error_lift[i])
    print('Nilai sebenarnya :',yv[i])
    print('Nilai numerik :',objective_lift(a[i],p1,p2,p3))
    print('#############################')
#print (yv[h-1])
#print (objective_lift(a[h-1],p1,p2,p3,p4,p5))

#WEIGHT
def objective_weight(x, q1, q2, q3):
	return (q1 * x) + ( q2 * x**2) + q3
pol_x = a
pol_y = yweight
#curve fit
pol, _ = curve_fit(objective_weight,pol_x,pol_y)
#summarize the parameter values
q1,q2,q3 = pol
print('y = %.5f * x + %.5f * x^2 + %.5f ' % (q1, q2, q3))
# plot input vs output
x=pol_x
y=pol_y
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(0, max(x), 1)
# calculate the output for the range
y_line = objective_weight(x_line, q1, q2, q3)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red',label = 'Curve Fitting Weight Result')
pyplot.legend(frameon= True, loc='best')
pyplot.show()

#ERROR WEIGHT
error_weight= [0 for i in range (h,-1,-1)]
for i in range (h,-1,-1):
    error_weight[i] = (yweight [i]- objective_weight(a[i],q1,q2,q3))/yweight [i]*100
    print(error_weight[i])
    print('Nilai sebenarnya :',yweight[i])
    print('Nilai numerik :',objective_weight(a[i],q1,q2,q3))
    print('#############################')

#print (yweight[h-1])
#"print (objective_weight(a[h-1],q1,q2,q3))

## ARAH SPANWISE
#Luas Area dibawah kurva lift metode curvefitting
def V_l(x):
    return (p1 * x) + (p2 * x**2) + p3
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
print (moment_lift)

#Luas Area dibawah kurva weight metode curvefitting
def V_w(x):
    return (q1 * x) + q2 
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
print (moment_weight)
