#t-distributed Stochastic Neighbor Embedding(t-SNE)
import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,\
    matthews_corrcoef,confusion_matrix,classification_report,roc_curve,auc,multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import label_binarize
from numpy import *
import matplotlib.pyplot as plt
from numpy import interp


def fold_test(model,train_x,label_train,name):
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    n_fold = 1
    train_x= np.array(train_x)
    label_train = np.array(label_train)
    label_train_y = label_binarize(label_train, classes=[1, 2, 0])
    n_classes = label_train_y.shape[1]
    for train_index, test_index in rs.split(train_x):
        X_train_fold, X_test_fold = train_x[train_index], train_x[test_index]
        y_train_fold, y_test_fold = label_train_y[train_index], label_train_y[test_index]
        classifier = OneVsRestClassifier(model, n_jobs=-1)
        classifier.fit(X_train_fold, y_train_fold)
        predicted_y = OneVsRestClassifier(model.predict(X_test_fold))
        #y_score = classifier.decision_function(X_test_fold)
        y_score = classifier.predict_proba(X_test_fold)
        outfile1 = open ("y_score.txt","w")
        outfile1.write("%s\t%s\t%s\t%s\n" %(str(y_score),str(y_test_fold),str(label_train),str(label_train_y)))
        outfile1.close()

        print(y_test_fold)
        print(y_score)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_fold[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        dataframe = pd.DataFrame(roc_auc,index=[0])
        dataframe.to_csv("./after_wgan_model/fungi/" + name + ".csv", mode='a', sep=',', index=False,
                             encoding="utf-8")

        # micro（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # 画图
        plt.figure(figsize=(9, 6))
        lw = 3
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        color = ['r', 'g', 'b']
        for i in range(3):
            if i == 0:
                plt.plot(fpr[i], tpr[i], color=color[i], lw=lw, label='ROC curve for healthy class (area = %0.2f)' % roc_auc[i])
            if i == 1:
                plt.plot(fpr[i], tpr[i], color=color[i], lw=lw, label='ROC curve for F_diseased class (area = %0.2f)' % roc_auc[i])
            if i == 2:
                plt.plot(fpr[i], tpr[i], color=color[i], lw=lw, label='ROC curve for V_diseased class (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 20}, weight='bold')

        plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 20}, weight='bold')

        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.weight"] = "bold"
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Multi-calss ROC (fungi_class_MLP)',fontdict={'family' : 'Times New Roman', 'size' : 20},weight='bold')
        plt.legend(loc='best')

        bwith = 2  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        ax.spines['top'].set_linewidth(bwith)  # 设置上‘脊梁’为红色
        ax.spines['right'].set_linewidth(bwith)  # 设置上‘脊梁’为无色
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.legend(loc='best')
        plt.savefig("./after_wgan_model/fungi/" + name + ".pdf", dpi=600, format="pdf")
        plt.show()

        n_fold += 1


def MLP(train_x,label_train,name,val_x3,label_val):
    outfile = open("./after_wgan_model/fungi/mlp_"+name+"_best_parameter.txt", "w")
    mlp = MLPClassifier()
    mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(100,80,70,50,40,30), (100,80,70,50,40,30,20)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "max_iter": [20],
                                 "verbose": [True],
                                 "activation": ["relu"],
                                 "alpha" : [1e-3,1e-5],
                                 }
    mlp_train = GridSearchCV(mlp, param_grid=mlp_clf__tuned_parameters, n_jobs=6)

    # mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 20), activation="relu",random_state=1)
    mlp_train.fit(train_x, label_train)
    best_parameters = mlp_train.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
        outfile.write("%s\t%s\n" % (str(para), str(val)))
    model_mlp = MLPClassifier(solver=best_parameters["solver"], alpha=best_parameters["alpha"],
                              hidden_layer_sizes=best_parameters["hidden_layer_sizes"],
                              activation=best_parameters["activation"],
                              verbose=best_parameters["verbose"],
                              max_iter=best_parameters["max_iter"],
                              random_state=1)
    model_mlp.fit(train_x, label_train)
    joblib.dump(model_mlp, "./after_wgan_model/fungi/model_mlp_" + name + ".pkl")
    fold_test(model_mlp, train_x, label_train, name)

    resultsTestingProb = model_mlp.predict_proba(val_x3)
    resultsTesting = []
    for indexResults in range(len(resultsTestingProb)):
        if float(resultsTestingProb[indexResults][1]) > 0.5 or float(resultsTestingProb[indexResults][1]) == 0.5:
            resultsTesting.append(1)
        else:
            resultsTesting.append(0)

    mccTesting = matthews_corrcoef(label_val, resultsTesting)
    print("MCC: " + str(mccTesting))
    aurocTesting = roc_auc_score(label_val, resultsTestingProb, multi_class="ovr")
    print("AUROC: " + str(aurocTesting))
    return

    # predict_results = mlp_model.predict(feature_test)
    # print(accuracy_score(predict_results, target_test))
    # conf_mat = confusion_matrix(target_test, predict_results)
    # print(conf_mat)
    # print(classification_report(target_test, predict_results))

    # sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’,
    # solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’,
    # learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,
    # tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    # early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    # n_iter_no_change=10)

    # 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
    # solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，
    # SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）,SGD标识随机梯度下降。


def model_training_randomforest(train_x,label_train,name,val_x3,label_val):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    outfile = open("./after_wgan_model/fungi/rfc_best_parameter.txt", "w")
    parameters = {'n_estimators': np.arange(1, 100, 10),
                  'max_depth': np.arange(1, 20, 2)}
    rfc = RandomForestClassifier(random_state=1)
    grid = GridSearchCV(rfc, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(train_x, label_train)
    best_parameters = grid.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
        outfile.write("%s\t%s\n"%(str(para),str(val)))
    model_rfc = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                       max_depth=best_parameters['max_depth'], random_state=1)
    model_rfc.fit(train_x,label_train)
    joblib.dump(model_rfc, "./after_wgan_model/fungi/"+name+"_model_rfc.pkl")
    fold_test(model_rfc, train_x, label_train, name)

    resultsTestingProb = model_rfc.predict_proba(val_x3)
    resultsTesting = []
    for indexResults in range(len(resultsTestingProb)):
        if float(resultsTestingProb[indexResults][1]) > 0.5 or float(resultsTestingProb[indexResults][1]) == 0.5:
            resultsTesting.append(1)
        else:
            resultsTesting.append(0)

    mccTesting = matthews_corrcoef(label_val, resultsTesting)
    print("MCC: " + str(mccTesting))
    aurocTesting = roc_auc_score(label_val, resultsTestingProb, multi_class="ovr")
    print("AUROC: " + str(aurocTesting))

    return model_rfc

def realneg_realpos(real_positive_data,real_negative_data):
    realneg_realpos = np.vstack((real_positive_data, real_negative_data))
    return realneg_realpos


if __name__ == "__main__":
    path1 = "./fungi/training_set/"
    path2 = "./fungi/validation_set/"
    path3 = "./fungi/independent_test_set/"
    path4 = "./after_wgan_model/fungi/Iteration_synthetic_fungi_class/"
    path5 = "./after_wgan_model/"
    real_positive = "fungi_class_training_positive.csv"
    fake_positive = "fungi_class_syn_positive.csv"
    real_negative = "fungi_class_training_negative.csv"
    real_positive_data = pd.read_csv(path1 + real_positive, index_col=0, header=0)
    real_positive_data2 = np.array(real_positive_data, dtype='float')
    real_positive_data3 = np.transpose(real_positive_data2)

    fake_positive_data = pd.read_csv(path4 + fake_positive, index_col=None, header=0)
    fake_positive_data2 = np.array(fake_positive_data, dtype='float')

    real_fake_positive_data = np.vstack((real_positive_data3, fake_positive_data2))

    real_negative_data = pd.read_csv(path1 + real_negative, index_col=0, header=0)

    real_negative_data2 = np.array(real_negative_data, dtype='float')
    real_negative_data3 = np.transpose(real_negative_data2)

    train_x = realneg_realpos (real_fake_positive_data,real_negative_data3)

    train_y = pd.read_csv(path5+"fungi_after_model_train_y.csv",index_col=0, header=None)
    group = train_y.iloc[:,0]
    group1 =list(group)
    label_train =[]
    for i in range(len(group1)):
        if group1[i] == "F_Disease":
            label_train.append(1)
        elif group1[i] == "V_Disease":
            label_train.append(2)
        elif group1[i] == "Health":
            label_train.append(0)


    val_x = pd.read_csv(path2+"fungi_class_validation.csv", index_col=0, header=0)
    val_x2 = np.array(val_x, dtype='float')
    val_x3 = np.transpose(val_x2)

    val_y = pd.read_csv(path2+"validation_data_id.csv", index_col=0, header=None)
    val_y2 = val_y.iloc[:, 0]
    val_y3 = list(val_y2)
    label_val = []
    for i in range(len(val_y3)):
        if val_y3[i] == "F_Disease":
            label_val.append(1)
        elif val_y3[i] == "V_Disease":
            label_val.append(2)
        elif val_y3[i] == "Health":
            label_val.append(0)


    test_x = pd.read_csv(path3+"fungi_class_independent_data.csv", index_col=0, header=0)

    test_y1 = pd.read_csv(path3+"indepent_data_id.csv", index_col=0, header=0)[0:]
    test_y2 = test_y1.iloc[:, 0]
    test_y3 = list(test_y2)
    label_indep = []
    for i in range(len(test_y3)):
        if test_y3[i] == "F_Disease":
            label_indep.append(1)
        elif test_y3[i] == "V_Disease":
            label_indep.append(2)
        elif test_y3[i] == "Health":
            label_indep.append(0)

    name= "MLP_fungi_class_realpos_realneg"

    MLP(train_x,label_train,name,val_x3,label_val)
    # model_training_randomforest(train_x,label_train,name,val_x3,label_val)
