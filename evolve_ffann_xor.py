import mga
import ffann
import numpy as np
import matplotlib.pyplot as plt

# ANN Params
nI = 2
nH1 = 50
nH2 = 10
nO = 1
WeightRange = 15
BiasRange = 15

# EA Params
popsize = 100
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.01 #0.05
demeSize = 2
generations = 50
tournaments = generations * popsize
boundaries = 0 #1

tar = [1,0,0,1]
inp = [[0,0],[0,1],[1,0],[1,1]]

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fit = 0.0
    for i in range(len(inp)):
        fit+=abs(tar[i]-nn.step(inp[i]))
    return (len(inp) - fit)/len(inp)

def sm_map(genotype):
    x_range = np.linspace(0,1, num=50)
    y_range = np.linspace(0,1, num=50)
    smm = np.zeros((50,50))
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    i=0
    for x in x_range:
        j=0
        for y in y_range:
            smm[i][j]=nn.step([x,y])
            j+=1
        i+=1
    return smm

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, 0, boundaries)
ga.run(tournaments)
ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()

smm = sm_map(bi)
plt.imshow(smm)
plt.show()
