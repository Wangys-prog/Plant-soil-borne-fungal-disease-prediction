import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,\
    matthews_corrcoef,confusion_matrix,classification_report,roc_curve,auc
import joblib
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
import pandas as pd




def MLP(train_x,label_train,test_x3,label_test,model1):
    outfile = open("./independent_testing/bacteria_mcc_auroc_independent_test_micro.txt", "w")
    #X_train = pd.DataFrame(data=train_x, columns=features_name)
    #Xval = pd.DataFrame(data=val_x3, columns=features_name)
    mlp_model = joblib.load(model1)
    mlp_model.fit(train_x, label_train)
    resultsTestingProb = mlp_model.predict_proba(test_x3)
    resultsTesting = []
    for indexResults in range(len(resultsTestingProb)):
        if float(resultsTestingProb[indexResults][1]) > 0.5 or float(resultsTestingProb[indexResults][1]) == 0.5:
            resultsTesting.append(1)
        else:
            resultsTesting.append(0)
    mccTesting = matthews_corrcoef(label_test, resultsTesting)
    aurocTesting = roc_auc_score(label_test, resultsTestingProb, multi_class="ovr")
    accuracy =accuracy_score(label_test, resultsTesting)
    precision= precision_score(y_true=label_test, y_pred=resultsTesting, average='micro')
    recall= recall_score(y_true=label_test, y_pred=resultsTesting, average='micro')
    f1= f1_score(label_test, resultsTesting, average='micro')
    print(accuracy,precision,recall,f1)
    outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n"%("mccTesting","aurocTesting","accuracy","precision","recall","f1"))
    outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (mccTesting, aurocTesting,accuracy,precision,recall,f1))
    return

def realneg_realpos(real_fake_positive_data, real_negative_data3):
    realneg_realpos = np.vstack((real_fake_positive_data, real_negative_data3))
    return realneg_realpos

if __name__ == "__main__":
    path1 = "./bacteria/training_set/"
    path2 = "./bacteria/validation_set/"
    path3 = "./bacteria/independent_test_set/"
    path4 = "./after_wgan_model/bacteria/Iteration_synthetic_bacteria_asv/"
    path5 = "./after_wgan_model/"
    real_positive = "bacteria_asv_training_positive0.05.csv"
    fake_positive = "bacteria_asv_syn_positive.csv"
    real_negative = "bacteria_asv_training_negative0.05.csv"
    real_positive_data = pd.read_csv(path1 + real_positive, index_col=0, header=0)
    real_positive_data2 = np.array(real_positive_data, dtype='float')
    real_positive_data3 = np.transpose(real_positive_data2)
    fake_positive_data = pd.read_csv(path4 + fake_positive, index_col=None, header=0)
    fake_positive_data2 = np.array(fake_positive_data, dtype='float')

    real_fake_positive_data = np.vstack((real_positive_data3, fake_positive_data2))

    real_negative_data = pd.read_csv(path1 + real_negative, index_col=0, header=0)
    real_negative_data2 = np.array(real_negative_data, dtype='float')
    real_negative_data3 = np.transpose(real_negative_data2)

    train_x = realneg_realpos(real_fake_positive_data, real_negative_data3)

    train_y = pd.read_csv(path5 + "bacteria_after_model_train_y.csv", index_col=0, header=None)
    group = train_y.iloc[:, 0]
    group1 = list(group)
    label_train = []
    for i in range(len(group1)):
        if group1[i] == "F_Disease":
            label_train.append(1)
        elif group1[i] == "V_Disease":
            label_train.append(2)
        elif group1[i] == "Health":
            label_train.append(0)

    test_x = pd.read_csv(path3 + "bacteria_asv_independent_data0.05.csv", index_col=0, header=0)
    test_x2 = np.array(test_x, dtype='float')
    test_x3 = np.transpose(test_x2)

    test_y = pd.read_csv(path3 + "indepent_data_id.csv", index_col=0, header=None)
    test_y2 = test_y.iloc[:, 0]
    test_y3 = list(test_y2)
    label_test = []
    for i in range(len(test_y3)):
        if test_y3[i] == "F_Disease":
            label_test.append(1)
        elif test_y3[i] == "V_Disease":
            label_test.append(2)
        elif test_y3[i] == "Health":
            label_test.append(0)

    model1 = open('../12.feature_selection/after_model/model_mlp_MLP_bacteria_asv_realpos_realneg.pkl', 'rb')
    #model_read = model1.read()

    features_name = real_negative_data.index


    MLP(train_x,label_train,test_x3,label_test,model1)