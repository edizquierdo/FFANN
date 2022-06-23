import mga
import ctrnn
import invpend
import numpy as np
import matplotlib.pyplot as plt
import sys

viz = int(sys.argv[1])
savedata = int(sys.argv[2])
number = int(sys.argv[3])
# viz=1
# savedata=0
# number=0

# ANN Params
n = 5
nI = 3
nO = 1
duration = 10
stepsize = 0.005
WeightRange = 15
BiasRange = 15
timeConstantMin = 0.025
timeConstantMax = 0.05

noisestd = 0.0

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
genesize = n*n + 2*n + nI*n + nO*n #n + 2*n + nI*n + nO*n
recombProb = 0.5
mutatProb = 0.05 #0.01
demeSize = 50
generations = 150 #50
boundaries = 1

# Fitness function
def fitnessFunction(genotype):
    nn = ctrnn.CTRNN(n,nI,nO)
    nn.setParameters(genotype,WeightRange,BiasRange,timeConstantMin,timeConstantMax)
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range:
        for theta_dot in thetadot_range:
            body.theta = theta
            body.theta_dot = theta_dot
            nn.initializeState(np.zeros(n))
            for t in time:
                nn.step(stepsize,body.state())
                f = body.step(stepsize, nn.out() + np.random.normal(0.0,noisestd))
                fit += f
    return fit/(duration*total_trials)

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = ctrnn.CTRNN(n,nI,nO)
    nn.setParameters(genotype,WeightRange,BiasRange,timeConstantMin,timeConstantMax)
    body = invpend.InvPendulum()
    fit = 0.0
    theta_hist=np.zeros((total_trials_exam,len(time))) #XX
    rep = 0
    kt = 0
    fitmap = np.zeros((len(theta_range_exam),len(thetadot_range_exam)))
    for theta in theta_range_exam:
        ktd = 0
        for theta_dot in thetadot_range_exam:
            k = 0 #XX
            body.theta = theta
            body.theta_dot = theta_dot
            nn.initializeState(np.zeros(n))
            indfit = 0.0
            for t in time:
                nn.step(stepsize,body.state())
                f = body.step(stepsize, nn.out())
                theta_hist[rep][k] = body.theta
                indfit += f
                k += 1
            fitmap[kt][ktd]=indfit
            ktd += 1
            fit += indfit
            rep += 1
        kt += 1
    print(fit/(duration*total_trials_exam))
    return theta_hist, fitmap

#def sm_map(genotype):
#    smm = np.zeros((len(theta_range_exam),len(thetadot_range_exam)))
#    nn = ctrnn.CTRNN(n,nI,nO)
#    nn.setParameters(genotype,WeightRange,BiasRange,timeConstantMin,timeConstantMax)
#    i=0
#    for theta in theta_range_exam:
#        j=0
#        for theta_dot in thetadot_range_exam:
#            nn.initializeState(np.zeros(n))
#            indfit = 0.0
#            for t in time:
#
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
