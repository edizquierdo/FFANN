import matplotlib.pyplot as plt
import numpy as np

reps = 8
generations = 100

dir = "F02"

af_list = np.zeros((reps,generations))
bf_list = np.zeros((reps,generations))
for k in range(reps):
    af=np.load(dir+"/avgfit"+str(k)+".npy")
    af_list[k]=af #(af+7.65)/7
    bf=np.load(dir+"/bestfit"+str(k)+".npy")
    bf_list[k]=bf #(bf+7.65)/7
    #bg=np.load(dir+"/bestgenotype"+str(k)+".npy")
    #sm=np.load("E07/smm"+str(k)+".npy")
    #th=np.load(dir+"/theta"+str(k)+".npy")
    #fm=np.load(dir+"/fitmap"+str(k)+".npy")

    # plt.plot(bf)
    # plt.plot(af)
    # plt.xlabel("Generations")
    # plt.ylabel("Fitness")
    # plt.title("Best and average fitness")
    # plt.show()
    #
    # plt.plot(th.T)
    # plt.show()
    #
    # #plt.imshow(sm)
    # #plt.show()
    #
    # print(np.mean(fm))
    #
    # plt.imshow(fm)
    # plt.show()

bf_mean = np.mean(bf_list,axis=0)
bf_std = np.std(bf_list,axis=0)/np.sqrt(reps)
af_mean = np.mean(af_list,axis=0)
af_std = np.std(af_list,axis=0)/np.sqrt(reps)

np.save("bf_mean_"+dir+".npy",bf_mean)
np.save("bf_std_"+dir+".npy",bf_std)
np.save("af_mean_"+dir+".npy",af_mean)
np.save("af_std_"+dir+".npy",af_std)

gens=range(generations)

plt.plot(gens,bf_mean)
plt.fill_between(gens,bf_mean-2*bf_std,bf_mean+2*bf_std,alpha=0.2)
plt.plot(gens,af_mean)
plt.fill_between(gens,af_mean-2*af_std,af_mean+2*af_std, alpha=0.2)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Best and average fitness")
plt.show()
#
# ######################
# gens=range(generations)
#
# bf_mean_E08=np.load("e_bf_mean_E08.npy")
# bf_std_E08=np.load("e_bf_std_E08.npy")
# af_mean_E08=np.load("e_af_mean_E08.npy")
# af_std_E08=np.load("e_af_std_E08.npy")
#
# bf_mean_f01=np.load("bf_mean_f01.npy")
# bf_std_f01=np.load("bf_std_f01.npy")
# af_mean_f01=np.load("af_mean_f01.npy")
# af_std_f01=np.load("af_std_f01.npy")
#
# bf_mean_f02=np.load("bf_mean_f02.npy")
# bf_std_f02=np.load("bf_std_f02.npy")
# af_mean_f02=np.load("af_mean_f02.npy")
# af_std_f02=np.load("af_std_f02.npy")
#
# bf_mean_f03=np.load("bf_mean_f03.npy")
# bf_std_f03=np.load("bf_std_f03.npy")
# af_mean_f03=np.load("af_mean_f03.npy")
# af_std_f03=np.load("af_std_f03.npy")
#
# bf_mean_f04=np.load("bf_mean_f04.npy")
# bf_std_f04=np.load("bf_std_f04.npy")
# af_mean_f04=np.load("af_mean_f04.npy")
# af_std_f04=np.load("af_std_f04.npy")
#
# bf_mean_f05=np.load("bf_mean_f05.npy")
# bf_std_f05=np.load("bf_std_f05.npy")
# af_mean_f05=np.load("af_mean_f05.npy")
# af_std_f05=np.load("af_std_f05.npy")
#
# plt.plot(gens,bf_mean_E08,label="FFANN")
# plt.fill_between(gens,bf_mean_E08-2*bf_std_E08,bf_mean_E08+2*bf_std_E08,alpha=0.2)
# plt.plot(gens,bf_mean_f01,label="CTRNN,d=0.05,t=0.1,m=0.01,R")
# plt.fill_between(gens,bf_mean_f01-2*bf_std_f01,bf_mean_f01+2*bf_std_f01,alpha=0.2)
# plt.plot(gens,bf_mean_f02,label="CTRNN,d=0.01,t=0.05,m=0.01,R")
# plt.fill_between(gens,bf_mean_f02-2*bf_std_f02,bf_mean_f02+2*bf_std_f02,alpha=0.2)
# plt.plot(gens,bf_mean_f03,label="CTRNN,d=0.005,t=0.05,m=0.05,R")
# plt.fill_between(gens,bf_mean_f03-2*bf_std_f03,bf_mean_f03+2*bf_std_f03,alpha=0.2)
# plt.plot(gens,bf_mean_f04,label="CTRNN,d=0.005,t=0.025,m=0.05,F")
# plt.fill_between(gens,bf_mean_f04-2*bf_std_f04,bf_mean_f04+2*bf_std_f04,alpha=0.2)
# plt.plot(gens,bf_mean_f05,label="CTRNN,d=0.005,t=0.025,m=0.05,R")
# plt.fill_between(gens,bf_mean_f05-2*bf_std_f05,bf_mean_f05+2*bf_std_f05,alpha=0.2)
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.title("Best Fitness (10 reps)")
# plt.legend()
# plt.show()
#
# ################
#
# # plt.fill_between(gens,bf_mean_local-2*bf_std_local,bf_mean_local+2*bf_std_local,alpha=0.2)
# # plt.plot(gens,af_mean_local)
# # plt.fill_between(gens,af_mean_local-2*af_std_local,af_mean_local+2*af_std_local, alpha=0.2)
# # plt.plot(gens,bf_mean_global)
# # plt.fill_between(gens,bf_mean_global-2*bf_std_global,bf_mean_global+2*bf_std_global,alpha=0.2)
# # plt.plot(gens,af_mean_global)
# # plt.fill_between(gens,af_mean_global-2*af_std_global,af_mean_global+2*af_std_global, alpha=0.2)
# # plt.xlabel("Generations")
# # plt.ylabel("Fitness")
# # plt.title("Best and average fitness")
# # plt.show()
