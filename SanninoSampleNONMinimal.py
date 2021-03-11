# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:11:52 2020

@author: Janek
"""
import matplotlib.pyplot as plt
import numpy as np
import Gauss
from openpyxl import load_workbook

M = 22*10**(-9);
Ksi = 1/6;
Delta = 0.1;
mu = 10**(-3)*M;
NF = 10;
t0=0                #chwila początkowa
f0=9.2*M
      #wartosc poczatkowa fi
                  #początkowa wartosc parametru hubble'a
N=500000       #liczba iteracji w równaniu
              #Stała częsc potencjału
laststep=0 #zapisywany jest w nim czas trwania inflacji
X=[]
              #Masa plancka
#def Hubble(V0,M,a,p,f):         #ewoucja parametru hubble'a
#   return(1/(np.sqrt(3)*M)*np.sqrt(np.abs(V0*(1-np.exp(-np.sqrt(2/3)*f/M))**2)))

Lambda = (Delta*16*(np.pi)**2)/19*(np.sqrt(20 +6*np.sqrt(23)) - np.sqrt(23) - 1);
P1=0.06751460649245225
P2=9.874406462565936
P3=-2.078822413171776
P4=-2.078822413171776
P5=10.503523771815289
P6=16.921875
P7=-8.3125
P8=1
P9=-16.9219
P10=-7.125 
P11=2
P12=1.1874999999999993
P13=1
P14=6
def step(H):                    #ewolucja korku czasowego
    return(1/(100*H))
def V(f):
    return (Lambda * f**4)/(4*NF**2 *(1+ (Ksi*f**2)/M**2)**2)*(f/mu)**(-16/19*Delta);

def df(f):
    return (-M**4 * Lambda * f**3 * (f/mu)**(-16/19*Delta)*(M**2 * (-19 + 4*Delta) + 4*Delta*Ksi*f**2)*np.sqrt((M**2 + Ksi*f**2)**2/(M**4 + (M**2*Ksi*(1 + 6*Ksi)*f**2)))/(19*NF**2*(M**2 + Ksi*f**2)**3))
def ddf(f):
    return (1/(6*NF**2)*(M**2 + Ksi*f**2)**3)*(M**2*Delta*f**2*(f/mu)**(-16*Delta/19))*(((M**2*(9.87441 - 2.07882*Delta)-2.07882*Delta*Ksi*f**2)*(M**2 + Ksi*(1 + 6*Ksi)*f**2)**2)/(Ksi**2 + np.sqrt(M**4 + M**2*Ksi*(1+6*Ksi)*f**2)/(M**2 + Ksi*f**2)**2) + (10.5035*(M**2 + Ksi*f**2)*(M**4*(16.9219 - 8.3125*Delta + 1*Delta**2) + M**2*(-16.9219 - 7.125*Delta + 2*Delta**2)*Ksi*f**2 + Delta *(1.1875 + 1*Delta)*Ksi**2*f**4))/(M**2 + Ksi*(1 + 6*Ksi)*f**2))
def epsilon(f):
    return (0.082048*M**10*(NF + (NF*Ksi*(f*M)**2)/M**2)**4*(M**2*(9.87441 - 2.07882*Delta)-2.07882*Delta*Ksi*(f*M)**2)**2)/(NF**4*f**2*(M**2 + Ksi*(f*M)**2)**4*(M**4 + M**2*Ksi*(1+6*Ksi)*(f*M)**2))
def eta(f):
    return(
        (P1*M**4*(NF+(NF*Ksi*(f)**2)/(M**2))**2*( (M**2*(P2+P3*Delta)+P4*Delta*Ksi*(f)**2)*(M**2+Ksi*(1 + 6*Ksi)*(f)**2)**2/(Ksi**2*np.sqrt(  (M**4+M**2*Ksi*(1 + 6*Ksi)*(f)**2)/(M**2+Ksi*(f)**2)**2    ))
         + (P5*(M**2+Ksi*(f)**2)*( M**4*(P6 + P7*Delta+P8*Delta**2)+M**2*(P9+P10*Delta +P11*Delta**2)*Ksi*(f)**2 +Delta*(P12+P13*Delta)*Ksi**2*(f)**4    )    )/(M**2+Ksi*(1+P14*Ksi)*(f)**2  )  ))/(NF**2*(f)**2*(M**2+Ksi*(f)**2)**3)
        )
def solve(f0,n,H0,X):       #główna funkcja programu, rozwiązanie numeryczne jak w Sample
    global En,laststep, t  # w E wsadzam wartosci epsilona, w t czas. laststep będzie wyswietlany jako czas inflacji. En jest potrzebne tylko do debugu i
    f=np.zeros(n+1)  #wartosć pola Langevin
    t=np.zeros(n+1) 
    H=np.zeros(n+1)  #Wartosc parametru Hubble'a
    fn=np.zeros(n+1) #wartosc pola slow-roll
    E=np.zeros(n+1) #tu wkładam epsilon 
    En=np.zeros(n+1) #tu wkładam epsilon bez perturbacji, aktualnie zbędne
    et=np.zeros(n+1) #tu wkładam etę 
    f[0]=f0
    fn[0]=f0
    H[0]=H0          #przyjmuję niefluktuującą wartosc parametru Hubble
    E[0]=epsilon(f0)
    et[0]=eta(f0)
    for i in range(n):      #Główna petla iteracyjna 
        if E[i]>= 1 or et[i]>=1:  #warunek konca inflacji
            f=f[:i]
            t=t[:i]
            H=H[:i]
            fn=fn[:i]           #Ucinam puste ogony list
            E=E[:i]
            laststep+=i
            et=et[:i]
            print(laststep/100)
            break #przerywam glówną pętle iteracji, gdy przynajmniej jeden z warunków ifa zostaje spelniony
        deviation=np.sqrt(((H[i])*32)/((2*np.pi)**2)*step(H[i])) #parametr sigma w gaussie
        s=Gauss.fluctuation(0,deviation) #fluktuacja- losowa liczba z rozkladu gaussa
        f[i+1]=f[i]-1/(3*H[i])*df(f[i])*step(H[i])+s*M #zdyskretyzowany Langevin
        fn[i+1]=fn[i]-1/(3*H[i])*df(fn[i])*step(H[i])        #zdyskretyzowany Slow-roll
        t[i+1]=t[i]+step(H[i]) #krok czasowy
        H[i+1]=H0#Hubble(V0,M,a,p,fn[i]) #stały/zmienny Huble
        E[i+1]=epsilon(fn[i])
        et[i+1]=eta(fn[i])
        X.append(H[i+1]*t[i+1])          # To wyciągamy z funkcji solve, Dłuugą listę z wypisanymi wszystkimi krokami czasowymi, podczas których trwała inflacja. Uwaga, lista ta jest globalna, z każdym wywołaniem funkcji solve będą dodawane do niej kolejne elementy. Wykorzystuję to w histogramowaniu
    fig, axs=plt.subplots(2,sharex=True)
    axs[0].plot(t*H0,f/M, color='black', linestyle='-', linewidth=1,label='fluctuating')
    axs[0].plot(t*H0,fn/M, color='green', linestyle='-', linewidth=1,label='slow-roll')   
    axs[0].set_title("Examplary field evolution, $\phi_0$={}".format(f0/M))
    axs[0].set(ylabel='$\phi$')
    #axs[0].set_ylim(0,2*f0)
    axs[1].plot(t*H0,E,color='blue',label= '$\epsilon$')
    #axs[1].set_ylim(E[2],np.max(E))
    axs[1].set_title("Slow-roll $\epsilon$ and $\eta$ parameter evolution")
    axs[1].set(xlabel='$Ht$')
    axs[1].hlines(1,0,laststep/100,colors='red',linestyle='--')
    axs[1].plot(t*H0,et,color='cyan',label=' $\eta$')
    axs[1].set(ylabel='S-R parameter values')
    fig.legend()
    for ax in axs.flat:
        ax.label_outer()
    #plt.xlabel('Ht')axs[0].set_xlim(0, 2)
    #plt.ylabel('$\phi$')
    #plt.grid(True)
    plt.savefig('Samplef0{}.png'.format(int(f0/M)))
    return[X]
H0=np.sqrt(np.abs(V(f0)/3))/M
solve(f0,N,H0,X)    