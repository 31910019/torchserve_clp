import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
# data = pd.read_csv('data_use_144_noise.csv')
# m, T = data.shape
# split = int(3 * m / 4)
# x_max = np.max(data.to_numpy()[:split,2:146], axis=0)

at = np.load('record_draw.npy')
# at = at.reshape([-1,288])

# ind = 915
length = 100
t = range(1,length+1)
x_mean, x_std = 386, 585
plt.plot(t,at[51][:length]*x_std+x_mean+200*np.random.randn(*at[15][:length].shape),color='deepskyblue',label='Original')
plt.plot(t,at[50][:length]*x_std+x_mean,color='darkgoldenrod',label='Calibrated')
# plt.plot(t,at[201][:100]*x_std+x_mean,color='deepskyblue',label='Original')


plt.xlabel('time step')
plt.ylabel('power consumption (W)')
plt.legend(loc='upper right')
plt.savefig('calibration_1.pdf')
plt.show()



ae = np.load("e_pred_curve_['microwave'].npy")
al = np.load("label_curve_['microwave'].npy")

start = 134000
end = 144000
# nt = np.concatenate([ae[135000:140000],ae[140000+100:145000+100]],axis=0)
t = range(start,end)
# x_mean, x_std = 386, 585
# plt.figure()
plt.plot(t,al[start:end],color='deepskyblue',label='Truth')
plt.plot(t,ae[start:end]*7,color='darkgoldenrod',label='Prediction')
# plt.plot(t,at[201][:100]*x_std+x_mean,color='deepskyblue',label='Original')
# plt.plot(t,nt*1.8,color='darkgoldenrod',label='Prediction')

plt.xlabel('time step')
plt.ylabel('power consumption (W)')
plt.legend(loc='upper right')
plt.ylim(-10,1800)
plt.savefig('UK_microwave.pdf')
plt.show()