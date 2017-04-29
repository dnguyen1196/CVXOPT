import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
from numpy.linalg import slogdet
from rank_nullspace import rank, nullspace


class OptimalPowerFlow(object):
    def __init__(self, AdmitMat, Active, Reactive, Voltage):
        # These are the constraints given by the system
        self.Y = AdmitMat
        self.P = Active
        self.Q = Reactive
        self.W = Voltage

        # Class variables
        self.COUNT = 4
        self.MAXITER = 1000
        self.M = 2  # number of inequality constraints
        self.EPS = 0.0001

    def test(self):
        self.concatenateCVector()
        self.generateF_matrices()

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

            self.F_MATRICES.insert(i, Psi_k)
            self.F_MATRICES.insert(i + 1, np.negative(Psi_k))

            self.F_MATRICES.insert(i + 2 * self.COUNT, Omg_k)
            self.F_MATRICES.insert(i + 1 + 2 * self.COUNT, np.negative(Omg_k))

            self.F_MATRICES.insert(i + 2 * self.COUNT, J_k)
            self.F_MATRICES.insert(i + 1 + 2 * self.COUNT, np.negative(J_k))

        self.print_test(self.F_MATRICES)

    def print_test(self, array):
        for i in array:
            print(i)

    # Perform the centering step, basically Newton's step on the SDP
    def center(self, y, t):
        ALPHA = 0.01
        BETA = 0.5
        EPS = 0.0001
        MAXITER = 1000
        MAX_INNER_ITER = 1000

        B = np.zeros(self.COUNT * 6, self.COUNT * 6)
        S = np.subtract(B, np.diag(y))

        for it in range(MAXITER):
            # Find the hessian
            Hess = self.calculateHessian(S)
            grad = self.calculateGradient(t, S)

            dY = solve(Hess, np.negative(grad))
            decrement = np.dot(np.transpose(grad), np.negative(dX))  # Calculate newton decrement
            if (decrement < EPS * 2):
                return y
            # Back tracking line search
            t = 1.0
            val = self.findCost(y, t)
            for i in range(MAX_INNER_ITER):
                y_1 = np.add(y, np.multiply(dY, t))
                val_1 = self.findCost(y_1, t)
                if (val_1 > val + ALPHA * t * np.dot(np.transpose(grad), dY)):  # Check stopping criterion
                    break
                t = BETA * t
            y = np.add(y, np.multiply(dY, t))  # Update y
        return y


        # Calulate the cost function or objective function

    def findCost(self, y, t):
        res = 0.0
        res -= t * np.dot(np.transpose(self.C), y)
        A = np.zeros(self.COUNT, self.COUNT)
        for i in range(len(y)):
            A += y[i] * self.F_MATRICES[i]
        A += np.identity(self.COUNT)
        # Find log determinant, there is an alternative calculation
        res -= slogdet(A)
        return res


        # Find the hessian at the point

    def calculateHessian(self, S):
        HESS = np.zeros((self.COUNT * 6, self.COUNT * 6))
        for i in range(self.COUNT * 6):
            for j in range(self.COUNT * 6):
                p1 = np.dot(inv(S), np.negative(self.F_MATRICES[i]))
                p2 = np.dot(inv(S), np.negative(self.F_MATRICES[j]))
                HESS[i][j] = np.trace(np.dot(p1, p2))
        return HESS

    def calculateGradient(self, t, S):
        grad = np.zeros(self.COUNT * 6)
        for i in range(len(grad)):
            A = np.dot(inv(S), np.negative(self.F_MATRICES[i]))
            grad[i] = t * self.C[i] + np.trace(A)
        return grad

        # Perform optimization and return the optimal y

    def optimizeY(self):
        # Generate all the F_matrices into an array
        self.generateF_matrices()
        # Use the barrier method
        y = np.zeros(self.COUNT)
        t = 1.0
        u = 150
        for i in range(1, MAXITER):
            y = self.center(y)
            if (self.M / t < self.EPS):
                return y
            t = u * t
        return y

        # Create the matrix A from the optimal
        # Extract optimal voltage as the nullspace of this matrix A

    def extractVoltage(self, y):
        # Construct matrix A
        A = np.zeros(self.COUNT, self.COUNT)
        for i in range(len(y)):
            A += y[i] * self.F_MATRICES[i]
        A += np.identity(self.COUNT)
        # Find the nullspace of matrix A
        ns = nullspace(A)
        return ns


        # Return the optimal voltage vector
        # Invokes all the above function

    def optimizePF(self):
        y = self.optimizeY()
        V = self.extractVoltage(y)
        # Do further checking that the null space is rank 1
        return V
