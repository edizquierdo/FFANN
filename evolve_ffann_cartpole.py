import mga                  #Optimimizer
import ffann             #Controller
import cartpole             #Task

import numpy as np
import matplotlib.pyplot as plt
import sys

# E01 --

viz = int(sys.argv[1])
savedata = int(sys.argv[2])
number = int(sys.argv[3])

# ANN Params
nI = 4
nH1 = 5
nH2 = 5
nO = 1
duration = 10
stepsize = 0.001
WeightRange = 15.0 #/nH1
BiasRange = 15.0 #/nH1

noisestd = 0.0 #0.01

# Fitness initialization ranges
trials_theta = 2
trials_thetadot = 2
trials_x = 2
trials_xdot = 2
total_trials = trials_theta*trials_thetadot*trials_x*trials_xdot
theta_range = np.linspace(-0.05, 0.05, num=trials_theta)
thetadot_range = np.linspace(-0.05, 0.05, num=trials_thetadot)
x_range = np.linspace(-0.05, 0.05, num=trials_x)
xdot_range = np.linspace(-0.05, 0.05, num=trials_xdot)

time = np.arange(0.0,duration,stepsize)

# EA Params
popsize = 50
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05
demeSize = popsize #2
generations = 100 #150
boundaries = 0 #1

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = cartpole.Cartpole()
    fit = 0.0
    for theta in theta_range:
        for theta_dot in thetadot_range:
            for x in x_range:
                for x_dot in xdot_range:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    for t in time:
                        nn.step(body.state())
                        f = body.step(stepsize, nn.output() + np.random.normal(0.0,noisestd))
                        fit += f
    return fit/(duration*total_trials)

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = cartpole.Cartpole()
    theta_hist=np.zeros((total_trials,len(time)))
    rep = 0
    for theta in theta_range:
        for theta_dot in thetadot_range:
            for x in x_range:
                for x_dot in xdot_range:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    k=0
                    for t in time:
                        nn.step(body.state())
                        f = body.step(stepsize, nn.output())
                        theta_hist[rep][k] = body.theta
                        k += 1
                    rep += 1
    return theta_hist

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()
theta_hist = evaluate(bi)

## Instead of plotting, save data to file
if viz:
    ga.showFitness()
    ga.showAge()
    plt.plot(theta_hist.T)
    plt.show()
    plt.show()
if savedata:
    np.save("bestfit"+str(number)+".npy",ga.bestHistory)
    np.save("avgfit"+str(number)+".npy",ga.avgHistory)
    np.save("theta"+str(number)+".npy",theta_hist)
    np.save("bestgenotype"+str(number)+".npy",bi)
