import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
from numpy.linalg import slogdet
from rank_nullspace import rank, nullspace
from numpy.linalg import cholesky

class OptimalPowerFlow(object):
    def __init__(self, AdmitMat, Active, Reactive, Voltage):
        # These are the constraints given by the system
        self.Y = AdmitMat
        self.P = Active
        self.Q = Reactive
        self.W = Voltage

        # Class variables
        self.COUNT = self.Y.shape[0]
        self.MAXITER = 10 # Set low for testing purposes
        self.M = 2  # number of inequality constraints
        self.EPS = 0.0001

    # Concatenate all constraints into a C vector of size 6n
    def concatenateCVector(self):
        P_low = self.P[:, 0]
        P_high = self.P[:, 1]
        Q_low = self.P[:, 0]
        Q_high = self.P[:, 1]
        W_low = self.W[:, 0]
        W_high = self.W[:, 1]
        self.C = np.concatenate((P_low, np.negative(P_high), Q_low, np.negative(Q_high), W_low, np.negative(W_high)))

    # This function generates the array of Fi matrices used during the Newton steps
    def generateF_matrices(self):
        self.F_MATRICES = [0] * (6 * self.COUNT)
        # Check these ranges, check the math
        for i in range(0, self.COUNT):
            e_k = np.zeros(self.COUNT)
            e_k[i] = 1

            J_k = np.outer(e_k, e_k)
            Y_k = np.dot(J_k, self.Y)
            Psi_k = np.divide(np.add(np.transpose(Y_k), Y_k), 2)
            Omg_k = np.divide(np.subtract(np.transpose(Y_k), Y_k), 2j)

            # Note the order -Psi Psi -Omg Omg -J J
            # Because the in dual is upper - lower so lower gets a "negative"
            # This is opposite to the sign pairing in concatenate vectors
            self.F_MATRICES[i] = np.negative(Psi_k)
            self.F_MATRICES[i + 1 * self.COUNT] = Psi_k

            self.F_MATRICES[i + 2 * self.COUNT] = np.negative(Omg_k)
            self.F_MATRICES[i + 3 * self.COUNT] = Omg_k

            self.F_MATRICES[i + 4 * self.COUNT] = np.negative(J_k)
            self.F_MATRICES[i + 5 * self.COUNT] = J_k


    def print_test(self, array):
        for i in array:
            print(i)

    # Perform the centering step, basically Newton's step on the SDP
    def center(self, y, t):
        EPS = 0.001

        for it in range(self.MAXITER):
            # Find the hessian and gradient
            Hess = self.calculateHessian(y)
            grad = self.calculateGradient(t, y)
            L = cholesky(Hess) # Making sure the hessian is always positive -> convex

            dY = solve(Hess, np.negative(grad))

            val_current = self.findCost(y+dY,t)
            val_underest = self.findCost(y,t) + np.transpose(grad,dY)
            print(val_current,"vs",val_underest)
            break
            decrement = np.dot(grad, np.negative(dY))  # Calculate newton decrement
            if decrement < EPS * 2:
                return y

            # Back tracking line search
            step = 1.0
            step = self.lineSearch(y,dY,step,t, grad)
            y = np.add(y, np.multiply(dY, step))  # Update y
        return y

    def lineSearch(self, y, dY, step, t, grad):
        ALPHA = 0.01
        BETA = 0.5
        val = self.findCost(y,t)
        y_1 = y + dY*step
        while np.any(y_1 > 0):  # Make sure y is always negative
            step *= BETA
            y_1 = y + dY * step

        for init in range(self.MAXITER):  # Line search steps
            val_1 = self.findCost(y_1, t)
            print (val_1, "vs", val, "vs", ALPHA * step * np.dot(grad, dY))
            if val_1 > val + ALPHA * step * np.dot(grad, dY):  # Check stopping criterion
                print("STOPPP!")
                break
            step *= BETA
        return step


    ''' Calulate the cost function or objective function
    '''
    def findCost(self, y, t):
        # Cost function is tc'y - log det(-sum... + I)
        res = t * np.dot(self.C, y)
        A = np.zeros((self.COUNT, self.COUNT),dtype=np.complex128)
        for i in range(len(y)):
            A -= y[i] * self.F_MATRICES[i]
        A += np.identity(self.COUNT)
        # Find log determinant, there is an alternative calculation
        (sign, logdet) = slogdet(A)
        res -= sign*logdet
        return res


    ''' Find the hessian at the point
    '''
    def calculateHessian(self, y):
        # Because of the "nice" constraint on diag(y) < 0
        # Hessian is imply (S^-1)^2
        B = np.zeros((self.COUNT * 6, self.COUNT * 6))
        S = np.subtract(B, np.diag(y))
        invS = inv(S)
        return np.dot(invS,invS)

    '''Find the gradient
    '''
    def calculateGradient(self, t, y):
        S = np.zeros((self.COUNT,self.COUNT),dtype=np.complex128)
        for i in range(len(y)):
            S -= y[i]*self.F_MATRICES[i]
        S += np.identity(self.COUNT)

        grad = np.zeros(self.COUNT * 6,dtype=np.complex128)
        invS = inv(S)
        for i in range(len(grad)):
            A = np.dot(invS, self.F_MATRICES[i])
            grad[i] = t * self.C[i] + np.trace(A)
        return grad

    '''Perform optimization and return the optimal y
    '''
    def optimizeY(self):
        self.concatenateCVector()
        self.generateF_matrices()
        # Use the barrier method
        # What is a feasible starting point? all 0s and all -1?
        y = np.negative(np.ones(6*self.COUNT))
        t = 1.0
        u = 150 # Varying this value gives a different convergence


        for i in range(1, self.MAXITER):
            y = self.center(y,t) # Perfom centering
            return y # For testing purposes
            if self.M / t < self.EPS:
                return y
            t = u * t
        return y

    '''Create the matrix A from the optimal
    Extract optimal voltage as the nullspace of this matrix A
    '''
    def extractVoltage(self, y):
        # Construct matrix A, work out the math more carefully
        A = np.zeros((self.COUNT, self.COUNT))
        for i in range(len(y)):
            A -= y[i] * self.F_MATRICES[i]
        A += np.identity(self.COUNT)
        # Find the null space of matrix A
        ns = nullspace(A)
        return ns


    '''Return the optimal voltage vector
    Invokes all the above function
    '''
    def optimize(self):
        y = self.optimizeY()
        V = self.extractVoltage(y)
        # Do further checking that the null space is rank 1
        return V
