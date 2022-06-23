import mga                  #Optimimizer
import ffann             #Controller
import inverted_pendulum    #Task

import numpy as np
import matplotlib.pyplot as plt
import sys

viz = int(sys.argv[1])
savedata = int(sys.argv[2])
number = int(sys.argv[3])

# ANN Params
nI = 3
nH1 = 5 #20
nH2 = 5 #10
nO = 1
duration = 10
stepsize = 0.05
WeightRange = 15.0 #/nH1
BiasRange = 15.0 #/nH1

noisestd = 0.0 #0.01

# Fitness initialization ranges
trials_theta = 6
trials_thetadot = 6
total_trials = trials_theta*trials_thetadot
theta_range = np.linspace(-np.pi, np.pi, num=trials_theta)
thetadot_range = np.linspace(-1.0,1.0, num=trials_thetadot)

trials_theta_exam = 50
trials_thetadot_exam = 50
theta_range_exam = np.linspace(-np.pi, np.pi, num=trials_theta_exam)
thetadot_range_exam = np.linspace(-1.0, 1.0, num=trials_thetadot_exam)
total_trials_exam = trials_theta_exam*trials_thetadot_exam

time = np.arange(0.0,duration,stepsize)

# EA Params
popsize = 50
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.01 #0.05
demeSize = popsize #2
generations = 150 #50
boundaries = 0 #1

# Fitness function
def fitnessFunction(genotype):
    nn = ff2hlann.ANN(nI,nH1,nH2,nO)
    #n.setParametersSTD(genotype,ParameterSTD)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = inverted_pendulum.InvPendulum()
    fit = 0.0
    for theta in theta_range:
        for theta_dot in thetadot_range:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time:
                nn.step(body.state())
                f = body.step(stepsize, nn.output() + np.random.normal(0.0,noisestd))
                fit += f
    return fit/(duration*total_trials)

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = ff2hlann.ANN(nI,nH1,nH2,nO)
    #n.setParametersSTD(genotype,ParameterSTD)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = inverted_pendulum.InvPendulum()
    fit = 0.0
    theta_hist=np.zeros((total_trials_exam,len(time))) #XX
    rep = 0 #XX
    kt = 0
    fitmap = np.zeros((len(theta_range_exam),len(thetadot_range_exam)))
    for theta in theta_range_exam:
        ktd = 0
        for theta_dot in thetadot_range_exam:
            k = 0 #XX
            body.theta = theta
            body.theta_dot = theta_dot
            indfit = 0.0
            for t in time:
                nn.step(body.state())
                f = body.step(stepsize, nn.output())
                theta_hist[rep][k] = body.theta #XX
                indfit += f
                k += 1 #XX
            fitmap[kt][ktd]=indfit
            ktd += 1
            fit += indfit
            rep += 1 #XX
        kt += 1
    print(fit/(duration*total_trials_exam))
    return theta_hist, fitmap

#def sm_map(genotype):
#    smm = np.zeros((len(theta_range_exam),len(thetadot_range_exam)))
#    nn = ff2hlann.ANN(nI,nH1,nH2,nO)
#    nn.setParameters(genotype,WeightRange,BiasRange)
#    i=0
#    for theta in theta_range_exam:
#        j=0
#        for theta_dot in thetadot_range_exam:
#            smm[i][j]=nn.step([np.cos(theta),np.sin(theta),theta_dot])
#            j+=1
#        i+=1
#    return smm

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()
theta_hist, fitmap = evaluate(bi)
#smm = sm_map(bi)

## Instead of plotting, save data to file
if viz:
    ga.showFitness()
    ga.showAge()
    plt.plot(theta_hist.T)
    plt.show()
#    plt.imshow(smm)
    plt.show()
if savedata:
    np.save("bestfit"+str(number)+".npy",ga.bestHistory)
    np.save("avgfit"+str(number)+".npy",ga.avgHistory)
    np.save("theta"+str(number)+".npy",theta_hist)
    np.save("fitmap"+str(number)+".npy",fitmap)
#    np.save("smm"+str(number)+".npy",smm)
    np.save("bestgenotype"+str(number)+".npy",bi)
