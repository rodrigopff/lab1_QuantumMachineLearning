# importing necessary packages
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.visualization import array_to_latex

#x = np.arange(0,8,1)    # define some vectors x and y
#y = x
#y = 2*x ## fazendo diferente aqui criiando u reta y=2x no luhar de y=x


np.random.seed(42)  # to make this code example reproducible
m = 8  # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
#print(X)
#print(X.shape)


# gera o vetor coluna com os parametros que será multiplicados pelos valores de cada caracteristica 
theta_row_vector = np.array([[3,5]])
theta_col_vector = theta_row_vector.T 

# Vetor coluna com o calculo de todas as instancias multiplicadas pelos parametros theta
X_ComParametrosMultiplicados = (X @ theta_col_vector) 

y =  4 + (X_ComParametrosMultiplicados) + np.random.randn(m, 1) 

y = y.T
x = X_ComParametrosMultiplicados.T



#print(np.concatenate((X_ComParametrosMultiplicados,y)))

#N = len(X)              
N = m              
nqubits = math.ceil(np.log2(N))    # compute how many qubits needed to encode either x or y

xnorm = np.linalg.norm(x)          # normalise vectors x and y
ynorm = np.linalg.norm(y)
x = x/xnorm
y = y/ynorm

print(y)
print(y[0])
print(y.shape)
print(x[0])
print(x.shape)

circ = QuantumCircuit(nqubits+1)   # create circuit
vec = np.concatenate((x[0],y[0]))/np.sqrt(2)    # concatenate x and y as above, with renormalisation

print(vec)
circ.initialize(vec, range(nqubits+1))
circ.h(nqubits)                    # apply hadamard to bottom qubit

circ.draw('mpl')                        # draw the circuit       



#Creates a quantum circuit to calculate the inner product between two normalised vectors

def inner_prod(vec1, vec2):
    #first check lengths are equal
    if len(vec1) != len(vec2):
        raise ValueError('Lengths of states are not equal')
        
    circ = QuantumCircuit(nqubits+1)
    vec = np.concatenate((vec1,vec2))/np.sqrt(2)
    
    circ.initialize(vec, range(nqubits+1))
    circ.h(nqubits)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)

    result = job.result()
    res = result.get_statevector(circ)
    o = np.real(res)
    

    m_sum = 0            
    for l in range(N):
        m_sum += o[l]**2         
        
    return 2*m_sum-1    


np.random.seed(42)  # to make this code example reproducible
m = 8  # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
#print(X)
#print(X.shape)


# gera o vetor coluna com os parametros que será multiplicados pelos valores de cada caracteristica 
theta_row_vector = np.array([[3,5]])
theta_col_vector = theta_row_vector.T 

# Vetor coluna com o calculo de todas as instancias multiplicadas pelos parametros theta
X_ComParametrosMultiplicados = (X @ theta_col_vector) 

y =  4 + (X_ComParametrosMultiplicados) + np.random.randn(m, 1) 

y = y.T
x = X_ComParametrosMultiplicados.T

N = m              
nqubits = math.ceil(np.log2(N))    # compute how many qubits needed to encode either x or y

xnorm = np.linalg.norm(x)          # normalise vectors x and y
ynorm = np.linalg.norm(y)
x = x/xnorm
y = y/ynorm

 
#x = np.arange(0,8,1)

#y = x
#y = 2*x
#print(y)

#N = len(x)
#nqubits = math.ceil(np.log2(N))
#xnorm = np.linalg.norm(x)
#ynorm = np.linalg.norm(y)
#x = x/xnorm
#y = y/ynorm
#print("######")
#print(ynorm)
#print(xnorm)
#print("######")

print("x: ", x)
print()
print("y: ", y)
print()
print("The inner product of x and y equals: ", inner_prod(x[0],y[0]))
#draw using latex
#inner_prod(x,y)[2].draw('latex')

#Implements the entire cost function by feeding the ansatz to the quantum circuit which computes inner products

def calculate_cost_function(parameters):

    a, b = parameters
    
    ansatz = a*x + b                        # compute ansatz
    ansatzNorm = np.linalg.norm(ansatz)     # normalise ansatz
    ansatz = ansatz/ansatzNorm
    
    y_ansatz = ansatzNorm/ynorm * inner_prod(y,ansatz)     # use quantum circuit to test ansatz
                                                           # note the normalisation factors
    return (1-y_ansatz)**2

x = np.arange(0,8,1)
y = 2*x

N = len(x)
nqubits = math.ceil(np.log2(N))
ynorm = np.linalg.norm(y)
y = y/ynorm

a = 1.0
b = 1.0
print("Cost function for a =", a, "and b =", b, "equals:", calculate_cost_function([a,b]))

#first set up the data sets x and y

x = np.arange(0,8,1)
y = 2*x   # + [random.uniform(-1,1) for p in range(8)]    # can add noise here
N = len(x)
nqubits = math.ceil(np.log2(N))
       
ynorm = np.linalg.norm(y)      # normalise the y data set
y = y/ynorm

x0 = [0.5,0.5]                 # initial guess for a and b

#now use different classical optimisers to see which one works best

out = minimize(calculate_cost_function, x0=x0, method="BFGS", options={'maxiter':200}, tol=1e-6)
out1 = minimize(calculate_cost_function, x0=x0, method="COBYLA", options={'maxiter':200}, tol=1e-6)
out2 = minimize(calculate_cost_function, x0=x0, method="Nelder-Mead", options={'maxiter':200}, tol=1e-6)
out3 = minimize(calculate_cost_function, x0=x0, method="CG", options={'maxiter':200}, tol=1e-6)
out4 = minimize(calculate_cost_function, x0=x0, method="trust-constr", options={'maxiter':200}, tol=1e-6)

out_a1 = out1['x'][0]
out_b1 = out1['x'][1]

out_a = out['x'][0]
out_b = out['x'][1]

out_a2 = out2['x'][0]
out_b2 = out2['x'][1]

out_a3 = out3['x'][0]
out_b3 = out3['x'][1]

out_a4 = out4['x'][0]
out_b4 = out4['x'][1]

plt.scatter(x,y*ynorm)
xfit = np.linspace(min(x), max(x), 100)
plt.plot(xfit, out_a*xfit+out_b, label='BFGS')
plt.plot(xfit, out_a1*xfit+out_b1, label='COBYLA')
plt.plot(xfit, out_a2*xfit+out_b2, label='Nelder-Mead')
plt.plot(xfit, out_a3*xfit+out_b3, label='CG')
plt.plot(xfit, out_a4*xfit+out_b4, label='trust-constr')
plt.legend()
plt.title("y = x")
plt.xlabel("x")
plt.ylabel("y")
plt.show()