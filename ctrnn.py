import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1-x))

class CTRNN():

    def __init__(self, size, inputsize, outputsize):
        self.Size = size                        # number of neurons in the network
        self.InputSize = inputsize
        self.OutputSize = outputsize
        self.Voltage = np.zeros(size)           # neuron activation vector
        self.TimeConstant = np.ones(size)       # time-constant vector
        self.Bias = np.zeros(size)              # bias vector
        self.Weights = np.zeros((size,size))     # weight matrix
        self.SensorWeights = np.zeros((inputsize,size))          # neuron output vector
        self.MotorWeights = np.zeros((size,outputsize))            # neuron output vector
        self.Output = np.zeros(size)            # neuron output vector
        self.Input = np.zeros(size)             # neuron output vector

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size))
        self.SensorWeights = np.random.uniform(-10,10,size=(self.Size))
        self.MotorWeights = np.random.uniform(-10,10,size=(self.Size))
        self.Bias = np.random.uniform(-10,10,size=(self.Size))
        self.TimeConstant = np.random.uniform(0.1,5.0,size=(self.Size))
        self.invTimeConstant = 1.0/self.TimeConstant

    def setParameters(self,genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax):
        k = 0
        for i in range(self.Size):
            for j in range(self.Size):                          # XXX
                self.Weights[i][j] = genotype[k]*WeightRange    # XXX
                k += 1
        for i in range(self.InputSize):
            for j in range(self.Size):
                self.SensorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            for j in range(self.OutputSize):
                self.MotorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            self.Bias[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.Size):
            self.TimeConstant[i] = ((genotype[k] + 1)/2)*(TimeConstMax-TimeConstMin) + TimeConstMin
            k += 1
        self.invTimeConstant = 1.0/self.TimeConstant

    def initializeState(self,v):
        self.Voltage = v
        self.Output = sigmoid(self.Voltage+self.Bias)

    def initializeOutput(self,o):
        self.Output = o
        self.Voltage = inv_sigmoid(o) - self.Bias

    def step(self,dt,i):
        self.Input = np.dot(self.SensorWeights.T, i) ###
        netinput = self.Input + np.dot(self.Weights.T, self.Output)
        self.Voltage += dt * (self.invTimeConstant*(-self.Voltage+netinput))
        self.Output = sigmoid(self.Voltage+self.Bias)

    def out(self):
        return np.dot(self.MotorWeights.T, self.Output)
