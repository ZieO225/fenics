""" brouillon """

from __future__ import print_function
from dolfin import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import boxfield_fixed as bx
from matplotlib import cm

import cv2 as cv



img_r = plt.imread('piece.png')
#plt.imshow(img_r)
#plt.show()

mu = 0
sigma = 0.01

img_r = img_r[:,:,0]
N = len(img_r)-1
noise = np.random.normal(mu, sigma, [N+1,N+1])
img = img_r + noise
print(" image bruitée ")

#plt.imshow(img)
#plt.show()

N = len(img)-1# pourque le nombre de noeud correspond avec le nombre de pixel

mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'Lagrange', 1)
x = mesh.coordinates().reshape((-1, 2))# cordonné du maillage


# cree le pas et construire un tableau (N+1*N+1,2)
h = 1./N
ii, jj = x[:, 0]/h, x[:, 1]/h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)

image_values = img[ii, jj]

# Créer un maillage et définir un espace fonctionnel

V = FunctionSpace(mesh, 'Lagrange', 1)

p_etoile = Function(V)
d2vm = dof_to_vertex_map(V)
image_values = image_values[d2vm] # reordonne par ordre croissante
p_etoile.vector()[:] = image_values
p_etoile = interpolate(p_etoile, V)


print('Plot p_etoile')

#plot(p_etoile)
#plt.show()



print('    CALCUL DE phi0')

# Définir la condition aux limites

def phi_0_boundary(x, on_boundary):

    return on_boundary

bc1 = DirichletBC(V, p_etoile, phi_0_boundary)


# Définir le problème variationnel

phi0 = TrialFunction(V)
v1 = TestFunction(V)
f1 = Constant(0.0)
a1 = dot(grad(phi0),grad(v1))*dx
L1 = f1*v1*dx

# Calculer la solution

phi0 = Function(V)
solve(a1 == L1, phi0, bc1)

print(phi0.vector()[:])

print(' plot phi0')

#dd = plot(phi0, mode='color')#,vmin=0,vmax=1)
#plt.colorbar(dd)
#plt.show()

""" calcul de new_phi """

print('    CALCUL DE new_phi0')


kappa  = Function(V)

kappa = 1/(grad(p_etoile)[0]*grad(p_etoile)[0] + grad(p_etoile)[1]*grad(p_etoile)[1]+0.001)

print('plot kappa')

#dd_ = plot(kappa, mode='color',vmin= 0,vmax=1)
#plt.colorbar(dd_)
#plt.show()

#condition aux litmite de Dirichlet

def phi_0_boundary(x, on_boundary):
    return on_boundary

bc2 = DirichletBC(V, p_etoile, phi_0_boundary)

# probleme variationnel

new_phi0 = TrialFunction(V)
v2 = TestFunction(V)
f2 = Constant(0.0)
a2 = kappa*dot(grad(new_phi0), grad(v2))*dx
L2 = f2*v2*dx

# Calcule de solution

new_phi0 = Function(V)
solve(a2 == L2, new_phi0, bc2)

print(new_phi0)
print(' plot new_phi0 ')
#ddd = plot(new_phi0, mode='color')#,vmin =0,vmax= 1)
#clb = plt.colorbar(ddd)
#clb.ax.set_title('new_phi0')
#plt.show()

""" calcule des phi ( fonction propres) """

print('    CALCUL DES FONCTONS PROPRES ')

phi_D = Constant(0.0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, phi_D, boundary)


# le probleme variationnel

phi = TrialFunction(V)
v = TestFunction(V) # fonction test
d = dot(Constant(1),v)*dx
a = dot(grad(phi), grad(v))*dx # forme billineaire
b = dot(phi,v)*dx # le second terme

# assemblage

asm = SystemAssembler(a,d,bc)
A = PETScMatrix()
asm.assemble(A)
asm = SystemAssembler(b,d)
B = PETScMatrix()
asm.assemble(B)
bc.zero(B)

# ajout des parametres pour assurer la convergence

solver = SLEPcEigenSolver(A,B)
solver.parameters["solver"] = "krylov-schur"
solver.parameters["spectrum"] = "target magnitude"
solver.parameters["problem_type"] = "gen_hermitian"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = 10.0


print('calcule des pi et P')




P = Function(V)
K = 200
solver.solve(K)
phi = Function(V)
P = phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    #print ('eigenvalue:', r)
    phi.vector()[:] = rx
    #d_phi = plot(phi, mode='color')#,vmin=0,vmax=1)
    #clb=plt.colorbar(d_phi)
    #clb.ax.set_title("phi_%s.png" %i)
    #plt.show()
    #plt.savefig("resultat/phi_%s.png" %i)
    p_i = assemble((p_etoile-phi0)*phi*dx) # clacule des pi
    P.vector()[:]  = P.vector()[:] + p_i*phi.vector()[:]

#d_P = plot(P, mode='color')#,vmin=0,vmax=1)
#clb=plt.colorbar(d_P)
#clb.ax.set_title("P.png" )
#plt.show()




print('    CALCUL DES FONCTIONS PROPRES AVEC LA MATRICE A')


new_phi = TrialFunction(V)
v3 = TestFunction(V)
d3 = dot(Constant(1),v3)*dx
a3 = kappa*dot(grad(new_phi), grad(v3))*dx

b3 = dot(new_phi, v3)*dx

# assemblage

new_asm = SystemAssembler(a3,d3,bc)
new_A = PETScMatrix()
new_asm.assemble(new_A)
new_asm = SystemAssembler(b3,d3)
new_B = PETScMatrix()
new_asm.assemble(new_B)
bc.zero(new_B)

solver = SLEPcEigenSolver(new_A,new_B)
solver.parameters["solver"] = "krylov-schur"
solver.parameters["spectrum"] = "target magnitude"
solver.parameters["problem_type"] = "gen_hermitian"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = 10.0

""" calacul des news_P """


P_1 = Function(V)
#K = 10
solver.solve(K)
new_phi = Function(V)
P_1 = new_phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    #print ('eigenvalue:', r)
    new_phi.vector()[:] = rx
    #d_new_phi = plot(new_phi, mode='color')#,vmin=0,vmax=1)
    #clb = plt.colorbar(d_new_phi)
    #clb.ax.set_title("new_phi0_%s.png" %i)
    #clb.ax.set_title('new_phi0_%s.png" %i')
    #plot(new_phi)
    #plt.show()
    #plt.savefig("resultat/new_phi_%s.png" %i)
    p_i = assemble((p_etoile-new_phi0)*new_phi*dx) # clacule des pi
    P_1.vector()[:]  = P_1.vector()[:] + p_i*new_phi.vector()[:]


d_P_1 = plot(P_1, mode='color')#,vmin=0,vmax=1)
clb=plt.colorbar(d_P_1)
#clb.ax.set_title("P_1.png")
plt.show()
