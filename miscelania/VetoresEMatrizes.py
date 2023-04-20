import numpy as np

# Este arquivo mostra algumas operacoes qeu devem ser efetuadas para prepara dados para um treinamento de um modelo 
# de regressao linear


## Mexendo no treinamento

# Gerando um dataset para o treinamento 

np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
# Cada coluna abaixo representa uma caracteristica. 
X = 2 * np.random.rand(m, 2)  # column vector 
#print(X)


# As linhas abaixo comentadas sao um exemplo de como pegar cada coluna do dataset gerado. 
#primeira_coluna = np.array([X[ :,0]]).T
#segunda_coluna = np.array([X[ :,1]]).T

#####################################################################################################################################
# Gerando um vetor de parametros theta que será usado na equacao de regressao linear 
# theta1 = 3 
# theta2 = 5
# y = tetha1 * x1 = theta2 * x2 ....
#######################################################################################################################################
theta_row_vector = np.array([[3,5]])
theta_col_vector = theta_row_vector.T # gera o vetor coluna com os parametros que será multiplicado pelos valores de cada caracteristica 
#print(theta_col_vector)

##########################################################################################################################################
## Gera um conjunto de targets para o dataset. 
# Utilizou-se aqui um equacao linear com dois parametros (theta1 e theta2) 
# yi = 4 + 3 * x1i  + 5 * x2i + ruido 
# com i representando cada linha do dataset ou seja cada instancia das duas caractersticas ( x1 e x2 )
# Para facilitar utilizamos a algebra linear evidentemente, que permite calcular o conjunto inteiro através de multiplicacao matricial. 
# pode se utlizar a funcao np.matmul ou o operador @ que indica uma multipliacao matricial.
# Quando soma-se um escalar a uma matriz como é feito abaixo  (paramereo tetha0 = 4 e o ruido np.random.randn(m, 1))
# cada linha da matriz resultante será somada destes valores. Desta forma nao precisamos coloca-los como colunas ou linhas das matrizes que 
# estão sendo multiplicadas.
############################################################################################################################################
#y =  4 + np.matmul(X, theta_col_vector) + np.random.randn(m, 1) 
y =  4 + (X @ theta_col_vector) + np.random.randn(m, 1) 
print(y)

#y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
#y = 4 + 3 * primeira_coluna  + 5 * segunda_coluna + np.random.randn(m, 1)  # column vector
#print(y)


