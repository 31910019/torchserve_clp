import json
import matplotlib.pyplot as plt

#[refrigerator, dishwaser,microwave,washer_dryer]
app = 'refrigerator'
data = 'redd_lf'

with open('./cnn_experiments/' +data + '/'+app+'/test_result.json') as f:
    raw_data = json.load(f)
if app=='dishwaser':
    app = 'dishwasher'

GT = []
PRED = []
x = []
for i in range(len(raw_data.get('gt'))):
    GT.append (raw_data.get('gt')[i][0])
    PRED.append (raw_data.get('pred')[i][0])


plt.rcParams['font.sans-serif']=['SimHei']
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(GT[7500:15000],color='r')
ax1.plot(PRED[7500:15000],color='g')
plt.legend(('真实值', '预测值'),loc='upper right',prop={"size":12})
plt.xlabel('采样点', size=18)
plt.ylabel('功率/W', size=18)
plt.title(app, size=18)
save_path = './cnn_result/'+data + '/'+app+'.jpg'
plt.savefig(save_path,)
plt.show()
