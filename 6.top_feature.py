from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,\
    matthews_corrcoef,confusion_matrix,classification_report,roc_curve,auc
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_classif,chi2
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.decomposition import PCA


def f_score(train_x,label_train,features_name):
    outputfile = open("./feature_selection/bacteria_feature_rank.txt","w")
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(train_x,label_train)
    lis1 = rf.feature_importances_
    result1 = [(x, y) for x, y in zip(features_name[1:], lis1)]
    result1 = sorted(result1, key=lambda x: x[1], reverse=True)
    feature_list = [x[0] for x in result1]
    for i in range(len(result1)):
        print(result1[i][0])
        outputfile.write("%s\t%s\n"%(str(result1[i][0]),str(result1[i][1])))
    outputfile.close()
    return feature_list


def MLP(train_x,train_y,feature_list,val_x3,label_val,model1):
    outfile = open("mcc_auroc_feature_selection.txt", "w")
    X_train = pd.DataFrame(data=train_x, columns=features_name)
    Xval = pd.DataFrame(data=val_x3, columns=features_name)
    mccTesting_list = []
    aurocTesting_list = []
    mlp_model = joblib.load(model1)
    for i in range(1,len(feature_list)):
        train_x_1 = X_train[feature_list[:i]]
        mlp_model.fit(train_x_1, train_y)
        resultsTestingProb = mlp_model.predict_proba(Xval[feature_list[:i]])
        resultsTesting = []
        for indexResults in range(len(resultsTestingProb)):
            if float(resultsTestingProb[indexResults][1]) > 0.5 or float(resultsTestingProb[indexResults][1]) == 0.5:
                resultsTesting.append(1)
            else:
                resultsTesting.append(0)
        mccTesting = matthews_corrcoef(label_val, resultsTesting)
        mccTesting_list.append(mccTesting)
        aurocTesting = roc_auc_score(label_val, resultsTestingProb, multi_class="ovr")
        aurocTesting_list.append(mccTesting)
        print(i,mccTesting)
        print(i,aurocTesting)
        outfile.write("%s\t%s\t%s\n" % (i, mccTesting, aurocTesting))
    dataframe1 = pd.DataFrame(mccTesting_list)
    dataframe1.to_csv("./12.feature_selection/mlp_bacteria_mcc_score_model2.csv",  sep=',', index=True,header=False, encoding="utf-8")

    max_index = mccTesting_list.index(max(mccTesting_list))
    feature_selection = train_x[feature_list[:max_index]]
    feature_selection_df = pd.DataFrame(feature_selection)
    feature_selection_df.to_csv("./12.feature_selection/mlp_bacteria_feature_selection_mccTesting_model2.csv",index=True, header=True, encoding="utf-8")

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

    val_x = pd.read_csv(path2 + "bacteria_asv_validation0.05.csv", index_col=0, header=0)
    val_x2 = np.array(val_x, dtype='float')
    val_x3 = np.transpose(val_x2)

    val_y = pd.read_csv(path2 + "validation_data_id.csv", index_col=0, header=None)
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


    model1 = open('../12.feature_selection/after_model/model_mlp_MLP_bacteria_asv_realpos_realneg.pkl', 'rb')
    #model_read = model1.read()

    features_name = real_negative_data.index


    feature_list = f_score(train_x,label_train,features_name)

    # MLP(train_x,label_train,feature_list,val_x3,label_val,model1)