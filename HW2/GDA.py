from math import pi, log

class GDA():
    """
    Constructor for GDA:
    @param pi: prior probability of the positive class
    @mu0: mean of the negative class
    @mu1: mean of the positive class
    @sigma: covariance matrix
    """
    def __init__(self):
        self.pi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None

    def train(self, x, y):
        '''
        estimate GDA parameters
        @param x:
        @param y:
        '''
        n = len(y)
        n0 = sum(1 for i in range(n) if y[i] == 0)
        n1 = n - n0

        self.pi = n0 / n
        self.mu0 = [0] * len(x[0])
        self.mu1 = [0] * len(x[0])

        for i in range(n):
            if y[i] == 0:
                for j in range(len(x[0])):
                    self.mu0[j] += x[i][j] / n0
            else:
                for j in range(len(x[0])):
                    self.mu1[j] += x[i][j] / n1

        self.sigma = [[0] * len(x[0]) for _ in range(len(x[0]))]

        for i in range(n):
            if y[i] == 0:
                for j in range(len(x[0])):
                    for k in range(len(x[0])):
                        self.sigma[j][k] += (x[i][j] - self.mu0[j]) * (x[i][k] - self.mu0[k])
            else:
                for j in range(len(x[0])):
                    for k in range(len(x[0])):
                        self.sigma[j][k] += (x[i][j] - self.mu1[j]) * (x[i][k] - self.mu1[k])

        for j in range(len(x[0])):
            for k in range(len(x[0])):
                self.sigma[j][k] /= n

    def predict(self, x):
        p0 = self.pi * self.gaussian(x, self.mu0, self.sigma)
        p1 = (1 - self.pi) * self.gaussian(x, self.mu1, self.sigma)
        return 0 if p0 > p1 else 1

    def gaussian(self, x, mu, sigma):
        det = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0]
        inv = [[sigma[1][1] / det, -sigma[0][1] / det], [-sigma[1][0] / det, sigma[0][0] / det]]
        diff = [x[i] - mu[i] for i in range(len(x))]
        exponent = -0.5 * (diff[0] * (diff[0] * inv[0][0] + diff[1] * inv[1][0]) + diff[1] * (diff[0] * inv[0][1] + diff[1] * inv[1][1]))
        return 1 / (2 * 3.14159265359 * (det ** 0.5)) * 2.71828182846 ** exponent

