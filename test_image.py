"""
resolution des fonctions propres du probleme elliptique :

nabla.(mu(x)nabla(phi)) = lambd*phi , pour tous dans (w U Ds)
phi = 0

"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import*   # importe les classes clés UnitSquare , FunctionSpace , Function ...

import boxfield_fixed as bx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from PIL import Image

# Creation du maillage , ici c'est u maillage carré

#mesh = UnitSquareMesh.create(256, 256, CellType.Type.quadrilateral )

""" Lecture de l image """

img = Image.open("man.png")
img.show()
img = np.array(img)/255
I = np.ravel(img)


""" essaie """

N = len(np.array(img))# pourque le nombre de noeud corespond avec le nombre de pixel
#print(N)
#print(N)

#mesh = UnitSquareMesh(N, N)
mesh = UnitSquareMesh.create(N, N, CellType.Type.quadrilateral )
Q_0 = FiniteElement("DG", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh,Q_0)
#x = mesh.coordinates().reshape((-1, 2))# cordonné du maillage

""" fin essaie """

# conditions limites
p_etoile = Function(V)
p_etoile.vector()[:] = I
p_etoile = interpolate(p_etoile, V)
print(p_etoile.vector()[:])


phi_D = Constant(0.0) #

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

#resolution des fonctions propres / ajout des parametres pour assurer la convergence

solver = SLEPcEigenSolver(A,B)
solver.parameters["solver"] = "krylov-schur"
solver.parameters["spectrum"] = "target magnitude"
solver.parameters["problem_type"] = "gen_hermitian"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = 5.5

K = 100
solver.solve(K)# calcule les K fonction
phi = Function(V)



#p_etoile = Expression("x[0]*x[0]*x[0] + x[1]*x[1]-1",degree=3)
#p_etoile = interpolate(p_etoile, V)


""" calcul de phi_0 """

def phi_0_boundary(x, on_boundary):

    return on_boundary

bc_1 = DirichletBC(V, p_etoile, phi_0_boundary)

#Define variational problem
phi_0 = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = inner(nabla_grad(phi_0), nabla_grad(v))*dx
L = f*v*dx


# Compute solution
phi_0 = Function(V)
solve(a == L, phi_0, bc_1)

print(phi_0.vector()[:])

""" calcule des pi et P """

#taille_phi = len(phi.vector()[:])


P = Function(V)
#p_i = Function(V)
#P_final =Function(V)
#p_0 = Function(V)
#P = phi_0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    phi.vector()[:] = rx
    #print(phi.vector()[:])
    #stoc_phi.vector()[i] = phi.vector()[:]
    p_i = assemble((p_etoile-phi_0)*phi*dx) # clacule des pi
    #P = P + p_i*phi
    P.vector()[:]  = P.vector()[:] + p_i*phi.vector()[:]


P.vector()[:] = phi_0.vector()[:] + P.vector()[:]
print(P.vector()[:])


""" plot 2D  de P_final, un problem avec le plot en 3D """

filename = "phi_0.pvd"
filename1 = "P.pvd"
filename2 = "p_etoile.pvd"
file = File ( filename)
file1 =  File (filename1)
file2 =  File (filename2)
file1 << P
file << phi_0
file2 << p_etoile
