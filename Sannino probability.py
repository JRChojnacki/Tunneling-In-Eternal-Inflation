# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:34:31 2020
@author: Jan Chojnacki
"""
import matplotlib.pyplot as plt    #Rysuneczki
import numpy as np   #Podstawowe operacje matematyczne, głównie wektory, macierze
import Gauss          #krótki moduł losujący liczbę z rozkładu Gaussa   
from openpyxl import load_workbook  #operacja na plikach w excelu
t0=0                #warunek początkowy czasu w równaniach różniczkowych
M= 1#2.435 * 10**18 #Masa Plancka, przyjęta za 1 oraz jej wartosć w GeV   
f0=4.75*M #warunek poczatkowy na wartosc pola w równaniach różniczkowych
N=1300   #liczba iteracji w równaniu, praktycznie dziala, jak czas przez jaki puszczamy ewolucje pola
X=[]          # baza histogramu, zmienna w którą wrzucam wartosć H*t, dla spełnionego warunku inflacji \phi>1
laststep=0 #zapisywany jest w nim czas trwania inflacji
k = 1    ##Platania
L = 1   ##Platania
a = 1.6 ##Platania
s = 1   ##Platania
m = 1   ##Platania
Va=1.90073*10**(-10)    ##Platania
def V(f):   #Definicja potencjału To trzeba zmienić w Sannino
    return(np.exp(-2* np.sqrt(2/3)
   *f* k) *(-3* a**4 + 192 *(-1 + np.exp(np.sqrt(2/3) *f* k))**2 - 
   3* a**2* (-16 + a**2 + 16 *np.exp(np.sqrt(2/3)* f* k)) + 128* L - 
   6* a**3* np.sqrt(np.abs(-16 + a**2 + 16 *np.exp(np.sqrt(2/3) *f* k)))* s - 
   4 *np.sqrt(2)*
     a* (-8 + a**2 + 8* np.exp(np.sqrt(2/3)* f* k) + 
      a *np.sqrt(np.abs(-16 + a**2 + 16* np.exp(np.sqrt(2/3)* f* k))) *s)**(3/2)) *Va)
def Hubble(V0,M,a,p,f):         #ewoucja parametru hubble'a, nie wykorzystana w programie, przyjmujemy H=H0 za stałą podczas inflacji
    return(1/(np.sqrt(3)*M)*np.sqrt(np.abs(V0*(1-np.exp(-np.sqrt(2/3)*f/M))**2)))
def step(H):                    #ewolucja korku czasowego, od Tudeliusa
    return(1/(100*H))
def df(f):  #Pierwsza pochodna potencjału RIP konwencja oznaczeń. To trzeba zmienić w Sannino
    return(-2* np.sqrt(2/3)* np.exp(-2* np.sqrt(2/3)* f* k)
   *k *(-3 *a**4 + 192 *(-1 + np.exp(np.sqrt(2/3)* f* k))**2 - 
    3 *a**2* (-16 + a**2 + 16* np.exp(np.sqrt(2/3) *f* k)) + 128* L - 
    6 *a**3* np.sqrt(np.abs(-16 + a**2 + 16 *np.exp(np.sqrt(2/3)* f* k))) *s - 
    4* np.sqrt(2)
      *a* (-8 + a**2 + 8 *np.exp(np.sqrt(2/3) *f *k) + 
       a* np.sqrt(np.abs(-16 + a**2 + 16 *np.exp(np.sqrt(2/3)* f* k))) *s)**(3/2)) *Va + 
 np.exp(-2* np.sqrt(2/3)*
    f* k)* (-16* np.sqrt(6)* a**2* np.exp(np.sqrt(2/3)* f* k)* k + 
    128* np.sqrt(6)* np.exp(np.sqrt(2/3)* f* k) *(-1 + np.exp(np.sqrt(2/3) *f *k)) *k - (
    16* np.sqrt(6)* a**3* np.exp(np.sqrt(2/3)* f* k)* k* s)/
    np.sqrt(np.abs(-16 + a**2 + 16* np.exp(np.sqrt(2/3)* f *k))) - 
    6* np.sqrt(2)*
      a* np.sqrt(np.abs(-8 + a**2 + 8 *np.exp(np.sqrt(2/3 )*f *k) + 
      a* np.sqrt(np.abs(-16 + a**2 + 16* np.exp(np.sqrt(2/3) *f *k)))
        *s)) *(8 *np.sqrt(2/3)* np.exp(np.sqrt(2/3)* f* k)* k + (
       8*np.sqrt(2/3)* a* np.exp(np.sqrt(2/3)* f* k)* k *s)/
       np.sqrt(np.abs(-16 + a**2 + 16* np.exp(np.sqrt(2/3)* f* k))))) *Va)
def ddf(f): #druga pochodna potencjału. To trzeba zmienić w Sannino, chyba że podajesz jawnie eta poniżej
    return(0)
def epsilon(dV,V):  #inflacyjny parametr epsilon<<1 => inflacja zachodzi To trzeba zmienić w Sannino
    return(0.5*(M*dV/V)**2)
def eta(ddV,V):     #inflacyjny parametr eta<<1 => inflacja zachodzi To trzeba zmienić w Sannino
    return(M**2 * ddV/V)
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
    for i in range(n):      #Główna petla iteracyjna 
        if E[i]>= 1 or et[i]>=1:  #warunek konca inflacji
            f=f[:i]
            t=t[:i]
            H=H[:i]
            fn=fn[:i]           #Ucinam puste ogony list
            E=E[:i]
            laststep+=i
            et=et[:i]
            #print(laststep)
            break #przerywam glówną pętle iteracji, gdy przynajmniej jeden z warunków ifa zostaje spelniony
        deviation=np.sqrt(((H[i])*32)/((2*np.pi)**2)*step(H[i])) #parametr sigma w gaussie
        s=Gauss.fluctuation(0,deviation) #fluktuacja- losowa liczba z rozkladu gaussa
        f[i+1]=f[i]-1/(3*H[i])*df(f[i])*step(H[i])+s #zdyskretyzowany Langevin
        fn[i+1]=fn[i]-1/(3*H[i])*df(fn[i])*step(H[i])        #zdyskretyzowany Slow-roll
        t[i+1]=t[i]+step(H[i]) #krok czasowy
        H[i+1]=H0#Hubble(V0,M,a,p,fn[i]) #stały/zmienny Huble
        v=V(f[i])    #
        dv=df(f[i])  #Te trzy parametry są potrzebne do wyznaczenia wartosci parametrow inflacyjnych eta, epsilon poniżej 
        ddv=ddf(f[i])#
        E[i+1]=epsilon(dv,v)
        et[i+1]=eta(ddv,v)
        X.append(H[i+1]*t[i+1])          # To wyciągamy z funkcji solve, Dłuugą listę z wypisanymi wszystkimi krokami czasowymi, podczas których trwała inflacja. Uwaga, lista ta jest globalna, z każdym wywołaniem funkcji solve będą dodawane do niej kolejne elementy. Wykorzystuję to w histogramowaniu
    return[X]
H0=np.sqrt(np.abs(V(f0)/3))/M #Wartosc parametru Hubble, wynika z drugiego rownania slow-rollu
for i in range(10000):      #10000 krotna ewolucja, pozwala na stworzenie statystyki i nabranie odpowiednio dużego sample do listy X[]
    solve(f0,N,H0,X)
    print(i/100)        #procent wykonania pętli całosci
B=[] #Tu zapisywane są czasy- biny poniższego histogramu.
C=[] #Tu zapisywane są zliczenia w danym binie- counts
counts,bins=np.histogram(X,1000,density=True) # Metoda np.histogram histogramuje listę X, tj. dzieli zakres najdłuzszego z czasów na 1000 podzbiorów i nalicza ile elementów z X jest w każdym z tych podzbiorów. Density=true normalizuje histogram do 1. Przypisanie bins,counts=(...) tworzy dwie listy. Counts to liczba zliczeń w danym przedziale czasowym bins
#plt.hist(X,200,range=(0,6),density=1) #rysuje histogram
for i in range(1000): #Tą pętlą przygotowuję dane do wykresu C(B). Z rozwiązania FOkkera-Plancka wiem, że prawdopodobieństwo zanika prawie eksponencjalnie, stąd -np.log
    if counts[i]!=0: #bez tego logarytm wybucha, jesli w danym binie nie trafi się zaden przypadek 
        C.append(-np.log(counts[i])) #
        B.append(bins[i])
lastbin=int(laststep/(N*10)) #Nie pamiętam po co, ale potrzebne
CL=C[:lastbin]
BL=B[:lastbin]
alpha,betha=np.polyfit(BL, CL, 1) #parametry fitu liniowego do danych, alpha to tutaj częstosc zaniku/decay rate/Gamma, jest najważniejszym wynikiem programu
Y=alpha*np.array(B)+betha #linia 

plt.plot(B,3*H0*np.array(B),linestyle='--',color='red',label='EI boundary')
plt.plot(B,Y,linewidth=2.0,color='green',label="Linear fit")
plt.plot(B, C, linewidth=1.0, color='black',label='Probability')   
plt.title(r'Probability that inflation occurs for the AS model, $\phi_0$={}'.format(f0)) #nazwa do zmiany w Sannino
plt.ylabel('-log(P)')
plt.xlabel('$Ht$')
plt.legend()

plt.savefig('ProbabilityPlataniaf0={}vareps.png'.format(f0)) #zapisuje obrazek, w nazwie jest wartosc początkowa pola. Warto zmienic nazwe

new_row_data = [   #niezbędne jesli chce sie miec automatyczny zapis w excelu 
    ["L= ",L,"alpha= ",a,"Gamma= ",alpha,'f0= ',f0,'3H0= ',3*H0],  #L i a nie występują w sannino, trzeba je zmienić na te odpowiednie parametry
    ]
wb = load_workbook("PlataniaNumerical.xlsx") #warto zmienic nazwe pliku w excelu
# Select First Worksheet
ws = wb.worksheets[0]
# Append 2 new Rows - Columns A - D
for row_data in new_row_data:
    # Append Row Values
    ws.append(row_data)
wb.save("PlataniaNumerical.xlsx")#trzeba zmienic nazwe, jak sie ruszało ją wyżej