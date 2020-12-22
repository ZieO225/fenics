""" brouillon """

from __future__ import print_function
from dolfin import *
from fenics import *

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib import cm


img = plt.imread('piece.png')
#plt.imshow(img)
#plt.show()

print(img.shape) #(960,1106,4)

img = img[:,:,0]

N = len(img)-1 # pourque le nombre de noeud correspond avec le nombre de pixel

mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'Lagrange', 1)
x = mesh.coordinates().reshape((-1, 2))# récuperation des cordonnées des noeuds du maillage

# creer le pas et construire un tableau (N+1*N+1,)
h = 1./N
ii, jj = x[:, 0]/h, x[:, 1]/h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)
image_values = img[ii, jj]

# définir un espace fonctionnel

V = FunctionSpace(mesh, 'Lagrange', 1)

p_etoile = Function(V)
d2vm = dof_to_vertex_map(V)
image_values = image_values[d2vm] # reordonne par ordre croissante
#print(image_values)

p_etoile.vector()[:] = image_values
p_etoile = interpolate(p_etoile, V)


#print('Plot p_etoile')

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


print(' plot phi0')

#dd = plot(phi0, mode='color')
#plt.colorbar(dd)
#plt.show()
kappa  = Function(V)

kappa = 1/((grad(p_etoile)[0]*grad(p_etoile)[0] + grad(p_etoile)[1]*grad(p_etoile)[1])+0.001)

print('plot kappa')

#dd_ = plot(kappa, mode='color',vmin= 0,vmax=1)
#plt.colorbar(dd_)
#plt.show()

""" calcul de new_phi """

print('    CALCUL DE new_phi0')

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
solve(a2 == L2,new_phi0, bc2)

print(new_phi0)

print(' plot new_phi0 ')
#ddd = plot(new_phi0, mode='color')
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

""" creation d'un solver et ajout de parametre pour assurer la convergeance """

solver = SLEPcEigenSolver(A,B) # cree EigenSolver
solver.parameters["solver"] = "krylov-schur"
solver.parameters["spectrum"] = "target magnitude"
solver.parameters["problem_type"] = "gen_hermitian"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = 10.0



P = Function(V)
K = 10
solver.solve(K)
phi = Function(V)
P = phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    phi.vector()[:] = rx
    #d_phi = plot(phi, mode='color')
    #clb=plt.colorbar(d_phi)
    #clb.ax.set_title("phi_%s.png" %i)
    #plt.show()
    p_i = assemble((p_etoile-phi0)*phi*dx)
    P.vector()[:]  = P.vector()[:] + p_i*phi.vector()[:]

#d_P = plot(P, mode='color')
#clb=plt.colorbar(d_P)
#clb.ax.set_title("P.png" )
#plt.show()




print("    CALCUL DES FONCTIONS PROPRES ET LA RECONSTRUCTION DE L'IMAGE AVEC kappa")


new_phi = TrialFunction(V)
v3 = TestFunction(V)
d3 = dot(Constant(1),v3)*dx
a3 = kappa*dot(grad(new_phi), grad(v3))*dx

b3 = dot(new_phi, v3)*dx


""" assemblage sur tout omega """

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
solver.solve(K)
new_phi = Function(V)
P_1 = new_phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    new_phi.vector()[:] = rx
    d_new_phi = plot(new_phi, mode='color')
    clb = plt.colorbar(d_new_phi)
    clb.ax.set_title("new_phi0_%s.png" %i)
    plot(new_phi)
    plt.show()
    p_i = assemble((p_etoile-new_phi0)*new_phi*dx)
    P_1.vector()[:]  = P_1.vector()[:] + p_i*new_phi.vector()[:]


""" plot de l'image reconstruire """

print("affichage de l'image reconstruire ")

plot(P_1)
d_P_1 = plot(P_1, mode='color')
clb=plt.colorbar(d_P_1)
#clb.ax.set_title("P_1.png")
plt.show()
