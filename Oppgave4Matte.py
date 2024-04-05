import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as lin

#Oppgave a)

# Opprette en diskretisering av intervallet [0,1] for x-aksen med m+2 punkter (inkluderer randpunkter)
m=20
x=np.linspace(0,1,m+2)
h=x[1]-x[0]

# Konstruere en differanseoperator L3 for den andre deriverte med hensyn på x
# samme ide som for Dirichlet, men nå med variabler tilsvarende randene
L3 = (1/h**2)*sp.diags([1,-2,1],[-1,0,1],shape=(m+2,m+2))

# vi må imidlertidig sette den første og siste rad lik null
# de vil bli erstatte av ligninger for den deriverte

# L3 blir ikke lenger diagonal, så datatypen må endres
L3 = sp.csr_matrix(L3) #(Laplace-matrisen)

# setter den første rad lik null
L3[0,0] = 0
L3[0,1] = 0
# setter den andre rad lik null
L3[-1,-1] = 0
L3[-1,-2] = 0

# identitetsmatrise, men også med første og siste rad satt lik null
# Identitetsmatrise som brukes i Kronecker-produktet, tilpasset grensebetingelsene
I3 = sp.eye(m+2)

I3s = sp.csr_matrix(I3)

I3s[0,0]=0
I3s[-1,-1]=0

# lager y-koordinatene

n=20

y=np.linspace(0,1,n+2)
k = y[1]-y[0]

# Gjør det samme for y-retningen

L4 = (1/k**2)*sp.diags([1,-2,1],[-1,0,1],shape=(n+2,n+2))

L4 = sp.csr_matrix(L4)
L4[0,0] = 0
L4[0,1] = 0
L4[-1,-1] = 0
L4[-1,-2] = 0

I4 = sp.eye(n+2)
I4s = sp.csr_matrix(I4)
I4s[0,0] = 0
I4s[-1,-1] = 0

# Vi lager matrisen fra kroneckerproduktet
# Kronecker-produkter for å konstruere Laplace-operatoren i 2D

B1 = sp.kron(L3,I4s) 
B2 = sp.kron(I3s,L4)

# Legg til randbetingelser i matrisa

# Følger koden fra 1d
# Neumann randbetingelser for x-aksen
# x-retning
NB3 = np.zeros((m+2,m+2))
# den deriverte, første rad
NB3[0,0] = -1/h #u0
NB3[0,1] = 1/h #u1
# den deriverte, siste rad
NB3[-1,-2] = -1/h
NB3[-1,-1] = 1/h

# Neumann randbetingelser for y-aksen
# y-retning
NB4 = np.zeros((n+2,n+2))
# den deriverte, første rad
NB4[0,0] = -1/k
NB4[0,1] = 1/k
# den deriverte, andre rad
NB4[-1,-2] = -1/k
NB4[-1,-1] = 1/k


# Lager matrisen fra kroneckerproduktet
# Fullstendige Neumann betingelser med bruk av Kronecker-produktet
NB1 = sp.kron(NB3,I4) 
NB2 = sp.kron(I3,NB4)

# Legger sammen ligningene fra interiøren og randbetingelsene
# Samlet operator for Poisson-problemet med Neumann randbetingelser
B = B1 + B2 + NB1 + NB2

#print(B.toarray())

# Randbetingelse i forcing

# en vektor (1,0,0,0,0....)
Nm_l = np.zeros(m+2)
Nm_l[0] = 1

# en vektor (0,0,...,0,0,1)
Nm_r = np.zeros(m+2)
Nm_r[-1] = 1

# en vektor (1,0,0,0,0....)
Nn_l = np.zeros(n+2)
Nn_l[0] = 1

# en vektor (0,0,...,0,0,1)
Nn_r = np.zeros(n+2)
Nn_r[-1] = 1

#inngang størrelse 0.2 plasseres mellom 0.4 og 0.6
# Funksjoner som definerer randbetingelser
# randbetingelser for u'(x,0)
def g1(x):
    if ( x > 0.3) and (x < 0.7):
        return (x-0.4)*(x-0.6)
    else:
        return 0

# randbetingelser for u'(x,1)
def g2(x):
    if (x > 0.3) and (x < 0.7):
        return 1
    else:
        return 0
    
# randbetingelser for u'(0,y)
def g3(y):
    return 0*y


# randbetingelser for u'(1,y)
def g4(y):
    return 0*y

g1= np.vectorize(g1)
g2=np.vectorize(g2)

# Anvende randbetingelsene
# bidrag fra u'(x,0) på vektoren G
G1 = sp.kron(g2(x),Nn_l) 
# bidrag fra u'(x,1) på vektoren G
G2 = sp.kron(g2(x),Nn_r) 
# bidrag fra u'(0,y) på vektoren G
G3 = sp.kron(Nm_l,g3(y)) 
# bidrag fra u'(1,y) på vektoren G
G4 = sp.kron(Nm_r,g4(y))

# setter sammen bidragene over
# Kombinere randbetingelser til en samlet kildevektor G
G = G1 + G2 + G3 + G4

G = G.toarray()

# Løs systemet

G = np.reshape(G,(m+2)*(n+2))

# Vi bruker en minstkvadraters løser, siden systemet har sannsynligvis uendelig mange løsninger
# Løse Poisson-problemet ved hjelp av minste kvadraters metode
v, istop, itn, r1norm = lin.lsqr(B, G)[:4]

# v er løsningen
# istop og itn forteller om iterasjoner brukt i løsningen
# r1norm er størrelse på feil - burde være nær 0 hvis systemet har løsninger

V = np.reshape(v,(m+2,n+2))

# lager rutenett for plotting
X, Y = np.meshgrid(x,y, indexing='ij')

# lager figuren
fig,ax2 = plt.subplots(subplot_kw ={"projection":"3d"}, figsize=(15,15))

# plotter løsningen V

ax2.plot_surface(X,Y, V) #vmin=Z.min() * 2, cmap=cm.Blues)

plt.show()

#oppgave b)

#beregn gradient
# Beregning av den deriverte av V
V1= np.diff(V, axis =0)
V2= np.diff(V, axis =1)

print(f"V1 = {V1}")
print(f"V2 ={V2}")

#oppgave c)
#Plotte gradienten av V som et vektorfelt
print(V1[:,0:-1].shape)
print(V2[0:-1,:].shape)

print(X[0:-1,0:-1].shape)

fig, ax = plt.subplots(figsize =(15,15))

ax.quiver(np.transpose(Y[0:-1,0:-1]), np.transpose(X[0:-1,0:-1]), V1[:,0:-1], V2[0:-1,:])

plt.show()
