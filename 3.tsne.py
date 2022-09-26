import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

syn_data = "Iteration_synthetic_bacteria_asv/Iteration_0_Synthetic_bacteria_asv_Training_Positive.txt"
orig_data = "bacteria_asv_training_positive.csv"

path = "./after_wgan_model/bacteria/"
#读取olddata
syndata1 = pd.read_csv(path+syn_data,index_col=None, header=None,sep=",").iloc[:, :-1]
#读取fakedata
origdata0 = pd.read_csv(path+orig_data, index_col=0, header=0)
#merge_real_fake_data

syndata = np.array(syndata1, dtype='float')
#syndata = np.transpose(syndata3)
print(syndata.shape)

origdata2 = np.array(origdata0, dtype='float')
origdata = np.transpose(origdata2)
print(origdata.shape)

old_new_data = np.vstack((syndata, origdata))

#定义标签
label = []
for rowIndex in range(len(syndata)):
    label.append(1)
for rowIndex in range(len(origdata)):
    label.append(0)
labelArray = np.asarray(label)

# tsne

tsne = manifold.TSNE(n_components=2, random_state=500)

old_new_tsne=tsne.fit_transform(old_new_data.data)

plt.figure(figsize=(9, 6))
for i in range(len(labelArray)):
    # if labelArray[i] == 0:
    #     s1 = plt.scatter(old_new_tsne[i, 0], old_new_tsne[i, 1], s=15, lw=3, color='r')
    if labelArray[i] == 1:
        s2 = plt.scatter(old_new_tsne[i, 0], old_new_tsne[i, 1], s=30, lw=3, color='g', marker='^')

plt.rcParams['font.sans-serif']=['Times New Roman']

plt.xticks(size=15,weight='bold')
plt.yticks(size=15,weight='bold')
plt.xlabel('x-tsne',fontdict={'family' : 'Times New Roman', 'size' : 20},weight='bold')

plt.ylabel('y-tsne',fontdict={'family' : 'Times New Roman', 'size' : 20},weight='bold')

# plt.title("Original data + Synthetic data (Bacteria_ASV_iteration=0)",fontdict={'family' : 'Times New Roman', 'size' : 20},weight='bold')
plt.title("Synthetic data",fontdict={'family' : 'Times New Roman', 'size' : 20},weight='bold')
# real (green dots) and synthetic (red dots) protein feature
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.weight"] = "bold"
# plt.legend((s1,s2),('Original samples','Synthetic samples') ,loc = 'best')
#plt.legend(s1, 'Original samples' ,loc = 'best')
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['top'].set_linewidth(bwith)  # 设置上‘脊梁’为红色
ax.spines['right'].set_linewidth(bwith)  # 设置上‘脊梁’为无色
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.savefig(path+"bacteria_ASV_0_syn_tsne_synthetic.pdf", dpi=600,format="pdf")

plt.show()




