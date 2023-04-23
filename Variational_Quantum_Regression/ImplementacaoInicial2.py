# Segunda versao dos testes para regressao linear com qiskit
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

####### PRIMEITO BLOCO ############
np.random.seed(42)  # to make this code example reproducible
m = 8  # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
print(X)
print(X.T)
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


circ.draw('mpl') 
plt.show()                       # draw the circuit
############ FIM PRIMEITO BLOCO ############


####### SEGUNDO BLOCO ############

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
    #job = execute(circ, backend, backend_options = {"zero_threshold": 1e-20})
    job = execute(circ, backend)

    result = job.result()
    o = np.real(result.get_statevector(circ))

    m_sum = 0
    for l in range(N):
        m_sum += o[l]**2
        
    return 2*m_sum-1

np.random.seed(42)  # to make this code example reproducible
m = 8  # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
print(X)
print(X.T)
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

print("x: ", x)
print()
print("y: ", y)
print()
print("The inner product of x and y equals: ", inner_prod(x[0],y[0]))

########## FIM SEGUNDO BLOCO ########################


########### TERCEIRO BLOCO ##########################

#Implements the entire cost function by feeding the ansatz to the quantum circuit which computes inner products

def calculate_cost_function(a,b):

    a = a.T    
    b = b
    
    ansatz = x @ a + b                        # compute ansatz
    ansatzNorm = np.linalg.norm(ansatz)     # normalise ansatz
    ansatz = ansatz/ansatzNorm
    #print(ansatz[:, 0])
    #print(y[:, 0])

  
    y_ansatz = ansatzNorm/ynorm * inner_prod(y[:, 0],ansatz)     # use quantum circuit to test ansatz
                                                           # note the normalisation factors
    return (1-y_ansatz)**2

np.random.seed(42)  # to make this code example reproducible
m = 8  # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
#print(X)
#print(X.T)
#print(X.shape)


# gera o vetor coluna com os parametros que será multiplicados pelos valores de cada caracteristica 
theta_row_vector = np.array([[3,5]])
theta_col_vector = theta_row_vector.T 

# Vetor coluna com o calculo de todas as instancias multiplicadas pelos parametros theta
X_ComParametrosMultiplicados = (X @ theta_col_vector) 

y =  4 + (X_ComParametrosMultiplicados) + np.random.randn(m, 1) 

#y = y.T
x = X
y = y
N = len(x)
nqubits = math.ceil(np.log2(N))
ynorm = np.linalg.norm(y)
y = y/ynorm

#a = np.array([[1,1]]).T
a = np.array([1,1])
b = 1.0
print("Cost function for a =", a, "and b =", b, "equals:", calculate_cost_function(a,b))
 
########### FIM TERCEIRO BLOCO ########################## 


############### QUARTO BLOCO ###############################

np.random.seed(42)  # to make this code example reproducible
m = 8 # number of instances
#X = 2 * np.random.rand(m, 1)  # column vector
X = 2 * np.random.rand(m, 2)  # column vector
#print(X)
#print(X.T)
#print(X.shape)


# gera o vetor coluna com os parametros que será multiplicados pelos valores de cada caracteristica 
theta_row_vector = np.array([[3,5]])
theta_col_vector = theta_row_vector.T 

# Vetor coluna com o calculo de todas as instancias multiplicadas pelos parametros theta
X_ComParametrosMultiplicados = (X @ theta_col_vector) 

y =  4 + (X_ComParametrosMultiplicados) + np.random.randn(m, 1) 

#y = y.T
x = X
y = y
N = len(x)
nqubits = math.ceil(np.log2(N))
ynorm = np.linalg.norm(y)
y = y/ynorm

#a = np.array([[1,1]]).T
b = 1

x0 = [0.5,0.5]                 # initial guess for a and b
#x0 = np.array([[1,1]])

#now use different classical optimisers to see which one works best

out = minimize(calculate_cost_function, x0=x0, args=(b,), method="BFGS", options={'maxiter':200}, tol=1e-6)
print(out) 