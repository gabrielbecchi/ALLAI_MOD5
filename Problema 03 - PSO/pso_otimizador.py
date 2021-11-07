from pyswarm import pso

def funcao(x):
	x1 = x[0]
	x2 = x[1]
	return x1+x2


lower_bound = [-10, -10]
upper_bound = [10, 10]

# MINIMIZADOR
xopt, fopt = pso(funcao, lower_bound, upper_bound, 
	swarmsize=100, maxiter=100,minstep=1e-4)
print(xopt)
print(fopt)
