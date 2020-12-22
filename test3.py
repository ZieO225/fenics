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

img = Image.open("snap.png")


img.show()
#img_reduce = tf.image.resize(img,(100,100)).numpy()
#img = np.sum(img,2)/3
#img.show()
img = np.array(img)
print(img)
exit()

N = len(img)-1# pourque le nombre de noeud corespond avec le nombre de pixel

print("bruité l'image")
mu = 0.1
sigma = 0.001
noise = np.random.normal(mu, sigma, [N+1,N+1])
img_b = img + noise
new_img = Image.fromarray(img_b) # convertir tableau en image
new_img.show()
new_img =  np.array(new_img)
#N = 500

print(" adaptation du maillage pour l'image nette")

mesh = UnitSquareMesh(N, N)
plot(mesh)
#plt.savefig("resultat/mesh.png")
plt.show()
plt.savefig("resultat/mesh.png")
exit()
x = mesh.coordinates().reshape((-1, 2))# cordonné du maillage
#exit()
# cree le pas et construire un tableau (N+1*N+1,2)
h = 1./N
ii, jj = x[:, 0]/h, x[:, 1]/h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)

image_values = img[ii, jj] # adapte les pixels aux noeuds
new_image_values = new_img[ii, jj]

V = FunctionSpace(mesh, 'Lagrange', 1)


d2vm = dof_to_vertex_map(V) #
#print(d2vm)
image_values = image_values[d2vm] # reordonne par ordre croissante
new_image_values = new_image_values[d2vm]
p_b = Function(V)
p_etoile = Function(V)
p_etoile.vector()[:] = image_values
p_b.vector()[:] = new_image_values
#p_etoilebis = Expression('((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) <= 0.2*0.2 + tol ? k_0:k_1' ,degree = 0, tol = 1E-14,k_0 = 1,k_1 = 0)
#p_etoile = Function(V)
#p_etoile = 1
#p_etoile = interpolate(p_etoilebis,V)
#plot(p_etoile)
print('Plot p_etoile')

dd = plot(p_etoile, mode='color')#,vmin=0,vmax=1)
clb = plt.colorbar(dd)
clb.ax.set_title('p_etoile')
plt.show()
print('Plot p__bruit')

dd_b = plot(p_b, mode='color')#,vmin=0,vmax=1)
clb = plt.colorbar(dd_b)
clb.ax.set_title('p_bruit')
plt.show()


print('    CALCUL DE phi0')

def phi_0_boundary(x, on_boundary):

    return on_boundary

bc1 = DirichletBC(V, p_b, phi_0_boundary)

#Define variational problem
phi0 = TrialFunction(V)
v1 = TestFunction(V)
f1 = Constant(0.0)
a1 = dot(grad(phi0),grad(v1))*dx
#a = dot(grad(phi0), grad(v))*dx
L1 = f1*v1*dx


# Compute solution
phi0 = Function(V)
solve(a1 == L1, phi0, bc1)

print(phi0.vector()[:])

print(' plot phi0')

dd = plot(phi0, mode='color')#,vmin=0,vmax=1)
plt.colorbar(dd)
plt.show()
#plt.savefig("resultat/phi0.png")

""" calcul de new_phi """

print('    CALCUL DE new_phi0')




kappa  = Function(V)
#gradP1 = grad(p_etoile)[0], V)
#gradP2 = grad(p_etoile)[1], V)
#norm_p_etoile = np.sum((gradP1.vector()[:])*(gradP1.vector()[:]) +  gradP2.vector()[:]*gradP2.vector()[:]) + epsilon
#norm_p_etoile = grad(p_etoile)[0])*grad(p_etoile)[0]) + grad(p_etoile)[1]*grad(p_etoile)[1]
kappa = 1/(grad(p_b)[0]*grad(p_b)[0] + grad(p_b)[1]*grad(p_b)[1]+ 0.001)
#kappa_ = 1/(grad(p_etoile)[0]*grad(p_etoile)[0] + grad(p_etoile)[1]*grad(p_etoile)[1]+ 0.001)

print('plot kappa')

dd_ = plot(kappa, mode='color')#,vmin= 0,vmax=1)
plt.colorbar(dd_)
plt.show()

def phi_0_boundary(x, on_boundary):
    return on_boundary

bc2 = DirichletBC(V, p_b, phi_0_boundary)

#Define variational problem
new_phi0 = TrialFunction(V)
v2 = TestFunction(V)
f2 = Constant(0.0)
#a2 = dot(float(1/norm_p_etoile)*grad(new_phi0), grad(v2))*dx # forme billineaire
a2 = kappa*dot(grad(new_phi0), grad(v2))*dx
L2 = f2*v2*dx
# Compute solution
new_phi0 = Function(V)
solve(a2 == L2, new_phi0, bc2)

print(new_phi0)
print(' plot new_phi0 ')
ddd = plot(new_phi0, mode='color')#,vmin =0,vmax= 1)
clb = plt.colorbar(ddd)
clb.ax.set_title('new_phi0')
plt.show()
print(new_phi0.vector()[:])

#plt.savefig("resultat/new_phi0.png")

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

#taille_phi = len(phi.vector()[:])


P = Function(V)
K = 5
solver.solve(K)
phi = Function(V)
P = phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    #print ('eigenvalue:', r)
    phi.vector()[:] = rx
    d_phi = plot(phi, mode='color')#,vmin=0,vmax=1)
    clb=plt.colorbar(d_phi)
    clb.ax.set_title("phi_%s.png" %i)
    #plot(phi)
    plt.show()
    #plt.savefig("resultat/phi_%s.png" %i)
    p_i = assemble((p_etoile-phi0)*phi*dx) # clacule des pi
    P.vector()[:]  = P.vector()[:] + p_i*phi.vector()[:]

d_P = plot(P, mode='color')#,vmin=0,vmax=1)
clb=plt.colorbar(d_P)
clb.ax.set_title("P.png" )
plt.show()



#print(np.mean(new_phi0.vector()[:]))
#print(np.mean(phi0.vector()[:]))




print('    CALCUL DES FONCTIONS PROPRES AVEC LA MATRICE A')


new_phi = TrialFunction(V)
v3 = TestFunction(V) # fonction test
#d = dot(Constant(1),v)*dx
#x1 = grad(P).vector()
d3 = dot(Constant(1),v3)*dx
#norm_p_etoile = (gradP1.vector()[:].norm('l2') +  gradP2.vector()[:].norm('l2'))
#norm_p = dot(grad(p_etoile),grad(p_etoile))
#a_1 = dot(float(1/norm_p_etoile)*grad(new_phi), grad(v))*dx # forme billineaire
a3 = kappa*dot(grad(new_phi), grad(v3))*dx

b3 = dot(new_phi, v3)*dx # le second terme

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
    d_new_phi = plot(new_phi, mode='color',vmin=0,vmax=1)
    clb = plt.colorbar(d_new_phi)
    clb.ax.set_title("new_phi0_%s.png" %i)
    #.ax.set_title('new_phi0_%s.png" %i')
    #plot(new_phi)
    plt.show()
    #plt.savefig("resultat/new_phi_%s.png" %i)
    p_i = assemble((p_b-new_phi0)*new_phi*dx) # clacule des pi
    P_1.vector()[:]  = P_1.vector()[:] + p_i*new_phi.vector()[:] #+ p_b.vector()[:]


d_P_1 = plot(P_1, mode='color')#,vmin=0,vmax=1)
clb=plt.colorbar(d_P_1)
clb.ax.set_title("P_1.png")
plt.show()

exit()
#print(P_1.vector()[:])
#plot(P_1)
plt.savefig("resultat/P_1.png")
p_etoile = interpolate(new_phi0, V)
P_box  = bx.FEniCSBoxField(p_etoile,(N,N))
P_ = P_box.values
X = 0; Y= 1

fig = plt.figure()
ax = fig.gca(projection='3d')
cv  = P_box.grid.coorv
surf = ax.plot_surface(cv[X], cv[Y],P_,cmap=cm.coolwarm,rstride=1,cstride=1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Visualisation 3D')
plt.rcParams["figure.figsize"] = [6,6]
plt.savefig("resultat/new_phi0.png")
