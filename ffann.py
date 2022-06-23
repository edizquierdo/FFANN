import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

class ANN:

    def __init__(self, NIU, NH1U, NH2U, NOU):
        self.nI = NIU                   # number of input units
        self.nH1 = NH1U                 # number of hidden units in layer 1
        self.nH2 = NH2U                 # number of hidden units in layer 2
        self.nO = NOU                   # number of output units
        # Weights
        self.wIH1 = np.zeros((NIU,NH1U))   # Weights that go from Input to Hidden Layer 1
        self.wH1H2 = np.zeros((NH1U,NH2U))
        self.wH2O = np.zeros((NH2U,NOU))
        # Biases
        self.bH1 = np.zeros(NH1U)       # Biases for hidden layer 1
        self.bH2 = np.zeros(NH2U)
        self.bO = np.zeros(NOU)
        # Activation
        self.Hidden1Activation = np.zeros(NH1U)
        self.Hidden2Activation = np.zeros(NH2U)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

    def setParametersSTD(self, genotype, std):
        k = 0
        for i in range(self.nI):
            for j in range(self.nH1):
                self.wIH1[i][j] = genotype[k]*(std/(self.nI*self.nH1))
                k += 1
        for i in range(self.nH1):
            for j in range(self.nH2):
                self.wH1H2[i][j] = genotype[k]*(std/(self.nH1*self.nH2))
                k += 1
        for i in range(self.nH2):
            for j in range(self.nO):
                self.wH2O[i][j] = genotype[k]*(std/(self.nH2*self.nO))
                k += 1
        for i in range(self.nH1):
            self.bH1[i] = genotype[k]*(std/self.nH1)
            k += 1
        for i in range(self.nH2):
            self.bH2[i] = genotype[k]*(std/self.nH2)
            k += 1
        for i in range(self.nO):
            self.bO[i] = genotype[k]*(std/self.nO)
            k += 1

    def setParameters(self, genotype, WeightRange, BiasRange):
        k = 0
        for i in range(self.nI):
            for j in range(self.nH1):
                self.wIH1[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.nH1):
            for j in range(self.nH2):
                self.wH1H2[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.nH2):
            for j in range(self.nO):
                self.wH2O[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.nH1):
            self.bH1[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.nH2):
            self.bH2[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.nO):
            self.bO[i] = genotype[k]*BiasRange
            k += 1

    def ablate_connection(self, i, j): # Set specific connection to 0
        pass

    def ablate(self, neuron): # Set outgoing connections to 0
        if (neuron<self.nI):
            i = neuron
            for j in range(self.nH1):
                self.wIH1[i][j] = 0.0
        if (neuron >= self.nI and neuron < self.nI+self.nH1):
            i = neuron-self.nI
            for j in range(self.nH2):
                self.wH1H2[i][j] = 0.0
        if (neuron >= self.nI+self.nH1 and neuron < self.nI+self.nH1+self.nH2):
            i = neuron-(self.nI+self.nH1)
            for j in range(self.nO):
                self.wH2O[i][j] = 0.0

    def information_lesion(self, neuron): # Fix state of neuron to avg of its state
        pass

    def step(self,Input):
        self.Input = np.array(Input)
        self.Hidden1Activation = relu(np.dot(self.Input.T,self.wIH1)+self.bH1)
        self.Hidden2Activation = relu(np.dot(self.Hidden1Activation,self.wH1H2)+self.bH2)
        self.OutputActivation = sigmoid(np.dot(self.Hidden2Activation,self.wH2O)+self.bO)
        return self.OutputActivation

    def output(self):
        return self.OutputActivation*2 - 1

    def states(self):
        return np.concatenate((self.Input,self.Hidden1Activation,self.Hidden2Activation,self.OutputActivation))
