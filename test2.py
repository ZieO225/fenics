from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sklearn
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter

img = Image.open("snap.png")
#img = np.sum(img,2)/3
#img = np.array(img)
#print(img[:,:,0])
#print(img[:,:,0] == np.sum(img,2)/3)
#img = np.array(img)
#print(img)
#print()
#img = ndimage.gausssian_filter(img, sigma=(1))
#img.show()

img = np.array(img)
print(img)
#print(img.shape)
model  = PCA(n_components = 225)
X = model.fit_transform(img)

#print(np.argmax(np.cumsum(model.explained_variance_ratio_))) # voir le nbre de variable à selectionner
model  = PCA(n_components = 153) # 71 variable selectionnées gardant 99% de leur varainces
X = model.fit_transform(img)
#print(X.shape)
#print(X)

#X = model.fit_transform(img)
#X1 = model.inverse_transform(X) # pca inverse
#print(X1[0])
plt.imshow(X)
plt.savefig("resultat2D/V1.png")
#print(X.shape)
#print(np.argmax(np.cumsum(model.explained_variance_ratio_)>0.99))

N = len(X)-1# pourque le nombre de noeud corespond avec le nombre de pixel
print(N)

mesh = UnitSquareMesh(N, N)
x = mesh.coordinates().reshape((-1, 2))# cordonné du maillage



# cree le pas et construire un tableau (N+1*N+1,2)
h = 1./N
ii, jj = x[:, 0]/h, x[:, 1]/h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)
#print(len(np.array(jj)))
#print(np.asarray(x).shape) # taille de x

#plot(mesh)
#plt.show()
#print("difference")
image_values = np.ravel(X) # adapte les pixels aux noeuds
#print(image_values)
V = FunctionSpace(mesh, 'Lagrange', 1)
p_etoile = Function(V)


d2vm = dof_to_vertex_map(V) #
image_values = image_values[d2vm] # reordonne par ordre croissante
print(image_values)
p_etoile.vector()[:] = image_values

""" calcule des phi ( fonction propres) """

phi_D = Constant(0.0) #

def boundary(x, on_boundary):
    return on_boundary
"""la fonction  on_boundary pour marquer la limite doit retourner une valeur booléenne : Vrai si la valeur donnée
 x se situe à la limite de Dirichlet et Faux autrement. L'argument on_boundary est fourni
par DOLFIN et est égal à True si x se trouve sur la limite physique du maillage. Dans le cas présent, où
nous sommes censés retourner Vrai pour tous les points de la frontière, nous pouvons simplement retourner la valeur fournie
de on_boundary . La fonction on_boundary sera appelée pour chaque point discret de la maille, qui
nous permet d'avoir des limites où u sont connus aussi à l'intérieur du domaine, si désiré.
On peut également omettre l'argument on_boundary, mais dans ce cas, nous devons tester la valeur de l'argument
coordonnées en x :
"""
bc = DirichletBC(V, phi_D, boundary)
""" phi_D est une instance contenant les valeurs phi_d, et boundary est une fonction (ou objet) décrivant
si un point se trouve sur la frontière où phi est spécifié."""

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



""" calcul de phi0 """

def phi_0_boundary(x, on_boundary):

    return on_boundary

bc_1 = DirichletBC(V, p_etoile, phi_0_boundary)

#Define variational problem
phi0 = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
#a = inner(nabla_grad(phi_0), nabla_grad(v))*dx
a = dot(grad(phi0), grad(v))*dx
L = f*v*dx


# Compute solution
phi0 = Function(V)
solve(a == L, phi0, bc_1)
plot(phi0)
plt.savefig("resultat2D/phi0.png")

#print(phi_0.vector()[:])

""" calcule des pi et P """

#taille_phi = len(phi.vector()[:])


P = Function(V)
K = 10
solver.solve(K)
phi = Function(V)
P = phi0
for i in range(K):
    r, c, rx, cx = solver.get_eigenpair(i)
    #print ('eigenvalue:', r)
    phi.vector()[:] = rx
    #plot(phi)
    #plt.savefig("resultat1/phi_%s.png" %i)
    p_i = assemble((p_etoile-phi0)*phi*dx) # clacule des pi
    P.vector()[:]  = P.vector()[:] + p_i*phi.vector()[:]


#print(P.vector()[:])
#plot(p_etoile)
#plt.show()
plot(P)
#print(P.vector()[:])
plt.savefig("resultat2D/P.png")
