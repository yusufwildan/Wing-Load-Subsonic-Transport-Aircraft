#Flight Critical modul
#Untuk menghitung gaya angkat dengan variasi (n,V)


CG_mac = float(input("Insert CG position in % MAC : "))
CG_pos = 13.67 + CG_mac/100*2.43 #Distance CG to nose

k = int(input('Insert the number of evaluating points : '))
n_inp=[0 for i in range (k)]
V_inp=[0 for i in range (k)]
for i in range (0,k):
    print ("poin (n,V) ke-",i+1)
    n_inp[i] = float(input("Insert n : "))
    V_inp[i] = float(input("Insert V (KEAS): "))
    print("##########################")

def titik(n,V):
    #INPUT LINE
    m = 27365.6 #Airplane mass (kg)
    W= m*9.81 #Airplane weight(N)
    po = 1.225 #density sea level1
    S= 67.527 # surface wing area in m^2
    St = 16.098#surface tail area in m^2
    AR = 12.30 # Wing Aspect Ratio 
    c = 2.423 #wing MAC
    ct = 1.863  #tailplane MAC 
    Cmo = 0.106  #wing moment coefficient
    Cmot = 0 #tailplane moment coefficient
    Mo = 0.5*po*V**2*S*c*Cmo #pitching moment
    Mot = 0.5*po*V**2*St*ct*Cmot #tailplane pitching moment
    d = 0.0 # distance drag to CG (m)
    t = 0.255 # distance trust to CG assumption(m)
    lw = (CG_mac-25)/100*c # distance Lw to CG (m)
    lt = 30.366+0.25*1.863-CG_pos # 30.366 is the distance of HTP leading edge from MAC ,distance Lt to CG (m)
    K = W*0.0 #W sin teta assumption
    ##########
    Lt = [0 for i in range (11)]
    Lw = [0 for i in range (11)]
    CD = [0 for i in range (11)]
    Clw = [0 for i in range (11)]
    D = [0 for i in range (11)]
    T = [0 for i in range (11)]
    for i in range (0,10):
        Lw[i] = abs(n)*W - Lt[i]
        Clw[i] = Lw[i]/(0.5*po*V**2*S)
        CD[i] = 0.02359*Clw[i]**2
        D[i] = 0.5*po*V**2*S*CD[i]
        T[i] = D[i]+K
        if (CG_mac>25):
            Lt[i+1]= (Lw[i]*lw +d*D[i]-t*T[i]-Mo-Mot)/lt
        else:
            Lt[i+1]= (-Lw[i]*lw -d*D[i]-t*T[i]-Mo-Mot)/-lt

    if (n>0):
        liftwing = Lw[9]
        lifttail = Lt[9]   

    else :
        liftwing = -Lw[9]
        lifttail = -Lt[9]
        
    
    print ('Untuk nilai n = ',n,' dan V = ', V,' maka diperoleh Lw = ' ,liftwing, ' N dan Lt = ', lifttail," N" )

for i in range (0,k):
    print(titik(n_inp[i],V_inp[i]))
    




