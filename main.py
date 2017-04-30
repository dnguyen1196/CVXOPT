import numpy as np 
from numpy.random import rand
from OPF import *

'''
Preprocess the data to extract 
1. Admittance matrix
2. Constraints on active powers
3. Constraints on reactive powers
4. Constraints on voltage


Invoke the class OPF that takes in all of these
'''

Admit_1_2 = 0.378-0.369j
Admit_2_4 = 0.341-0.23j
Admit_2_3 = 0.437-0.427j

y = rand(4)/10
Y = np.zeros((4,4),dtype=np.complex_)

Y[0][0] = y[0] + Admit_1_2
Y[0][1] = -np.negative(Admit_1_2)
Y[0][2] = Y[0][3] = 0

Y[1][0] = -Admit_1_2
Y[1][1] = y[1] + Admit_1_2 + Admit_2_3 + Admit_2_4
Y[1][2] = -Admit_2_3
Y[1][3] = -Admit_2_4

Y[2][0] = Y[2][3] = 0
Y[2][2] = y[2] + Admit_2_3
Y[2][1] = -Admit_2_3

Y[3][0] = Y[3][2] = 0
Y[3][3] = y[3] + Admit_2_4
Y[3][1] = -Admit_2_4

ACTIVE = np.array([0,44.98,71.41,71.41])
REACTIVE = np.array([0,44.1,70,70])

ACTIVE_CONSTRAINTS = np.zeros([4,2])
ACTIVE_CONSTRAINTS[:,0] = [0,39,65,65]
ACTIVE_CONSTRAINTS[:,1] = [6,50,77,77]

REACTIVE_CONSTRAINTS = np.zeros([4,2])
REACTIVE_CONSTRAINTS[:,0] = [0,39,64,64]
REACTIVE_CONSTRAINTS[:,1] = [6,50,77,77]

VOLTAGE_CONSTRAINTS = np.zeros([4,2]) # Vanja says this is legit
VOLTAGE_CONSTRAINTS[:,0] = np.array([0.9,0.9,0.9,0.9])
VOLTAGE_CONSTRAINTS[:,1] = np.array([1.1,1.1,1.1,1.1])

# Make sure that voltage constraints is not negative!!

OPF = OptimalPowerFlow(Y, ACTIVE_CONSTRAINTS, REACTIVE_CONSTRAINTS, VOLTAGE_CONSTRAINTS)

OPF.concatenateCVector()
OPF.generateF_matrices()
y = np.negative(np.ones(6 * OPF.COUNT))
t = 1.0

y = OPF.center(y,t)
# print("Center:",y)
