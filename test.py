from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import*   # importe les classes clés UnitSquare , FunctionSpace , Function ...

# Create mesh and define function space
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, "Lagrange", 1)
# Define boundary conditions
phi_0 = Expression("1")
def phi_0_boundary(x, on_boundary):
return on_boundary
bc = DirichletBC(V, phi_0, phi_0_boundary)
#Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx


# Compute solution
u = Function(V)
solve(a == L, u, bc)
# Plot solution and mesh
plot(u)
plot(mesh)
# Dump solution to file in VTK format
file = File(".pvd")
file << u
# Hold plot
interactive()
