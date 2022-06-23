import ffann                #Controller
import invpend              #Task
import cartpole             #Task

import numpy as np
import matplotlib.pyplot as plt

# ANN Params
nI = 3+4
nH1 = 5
nH2 = 5
nO = 1
WeightRange = 15.0
BiasRange = 15.0

noisestd = 0.0

# Task Params
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 10
stepsize_CP = 0.001
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)

# Fitness initialization ranges
trials_theta_IP = 6
trials_thetadot_IP = 6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

trials_theta_CP = 2
trials_thetadot_CP = 2
trials_x_CP = 2
trials_xdot_CP = 2
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

def record_traces(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    # Task 1
    ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
    theta_hist_ip = np.zeros((total_trials_IP,len(time_IP)+1))
    body = invpend.InvPendulum()
    k = 0
    fit = 0.0
    trial = 0
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            theta_hist_ip[trial][0]=body.theta
            timestep = 1
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4))))
                ip[k] = nn.states()
                f = body.step(stepsize_IP, nn.output() + np.random.normal(0.0,noisestd))
                theta_hist_ip[trial][timestep] = body.theta
                k += 1
                fit += f
                timestep += 1
            trial += 1
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    # Task 2
    cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
    theta_hist_cp = np.zeros((total_trials_CP,len(time_CP)+1))
    body = cartpole.Cartpole()
    k=0
    fit = 0.0
    trial = 0
    for theta in theta_range_CP:
        for theta_dot in thetadot_range_CP:
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    theta_hist_cp[trial][0]=body.theta
                    timestep = 1
                    for t in time_CP:
                        nn.step(np.concatenate((np.zeros(3),body.state())))
                        cp[k] = nn.states()
                        f = body.step(stepsize_CP, nn.output() + np.random.normal(0.0,noisestd))
                        theta_hist_cp[trial][timestep] = body.theta
                        fit += f
                        k += 1
                        timestep += 1
                    trial += 1
    fitness2 = fit/(duration_CP*total_trials_CP)
    print(fitness1,fitness2,fitness1*fitness2)
    return ip,cp,theta_hist_ip,theta_hist_cp

for ind in range(10):
   bi = np.load("EF02/bestgenotype"+str(ind)+".npy")
   ip,cp,tip,tcp = record_traces(bi)
   # plt.plot(tip.T)
   # plt.show()
   # plt.plot(tcp.T)
   # plt.show()
   # np.save("cp_neuraltraces_"+str(ind)+".npy",cp)
   # np.save("ip_neuraltraces_"+str(ind)+".npy",ip)
   # np.save("cp_theta_"+str(ind)+".npy",tcp)
   # np.save("ip_theta_"+str(ind)+".npy",tip)

# ind=9
# bi = np.load("EF01/bestgenotype"+str(ind)+".npy")
# ip,cp = record(bi)
# for hidden in range(10):
#     for inp in range(3):
#         plt.plot(ip.T[inp],ip.T[7+hidden],'.')
#     plt.show()
# for hidden in range(10):
#     for inp in range(3,7):
#         plt.plot(cp.T[inp],cp.T[7+hidden],'x')
#     plt.show()
