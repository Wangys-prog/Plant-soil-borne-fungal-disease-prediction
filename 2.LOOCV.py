from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

def LOOCV(train_x,train_y,model):
    prediction_list = []
    real_list = []
    loo = LeaveOneOut()
    loo.get_n_splits(train_x)
    for train_index, test_index in loo.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        # knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        model_train = model.fit (X_train, y_train)
        predicted_y = model_train.predict(X_test)
        prediction_list.append(predicted_y)
        real_list.append(y_test)
    accuracy = accuracy_score(real_list, prediction_list)
    return accuracy

    #recall = recall_score(real_list, prediction_list, average='weighted')


def realpos_fakepos(real_positive,fake_positive):
    realpos_fakepos = np.vstack((real_positive, fake_positive))
    # realneg_realpos_scaler = StandardScaler().fit_transform(realneg_realpos)
    label = []
    for rowIndex in range(len(real_positive)):
        label.append(1)
    for rowIndex in range(len(fake_positive)):
        label.append(0)
    labelArray1 = np.asarray(label)
    return realpos_fakepos,labelArray1

def knn_model(train_x,train_y):
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    #knn_model.fit(train_x, train_y)
    #joblib.dump(knn_model, "合成model_knn5.pkl")
    #cross_score(knn_model, train_x, train_y,"knn")
    accuracy = LOOCV(train_x, train_y, knn_model)
    return accuracy


if __name__ == "__main__":

    real_positive = "./after_wgan_model/bacteria/bacteria_species_training_positive.csv"
    path = "./after_wgan_model/bacteria/Iteration_synthetic_bacteria_species/"
    outfile = open("./after_wgan_model/bacteria/bacteria_species_wgan_loocv.txt","a")
    outfile.write("Iteration"+"\t"+"bacteria_species_loocv"+"\n")
    real_positive_data = pd.read_csv(real_positive, index_col=0, header=0)
    real_positive_data2 = np.array(real_positive_data, dtype='float')
    real_positive_data3 = np.transpose(real_positive_data2)
    for iteration in range(10000):
        if iteration % 200 == 0:
            print(iteration)
            fake_positive = path+"Iteration_"+ str(iteration) +"_Synthetic_bacteria_species_Training_Positive.txt"
            fake_positive_data = pd.read_table(fake_positive, sep=",", header=None, index_col=None)
            fake_positive_data2 = fake_positive_data.drop(fake_positive_data.columns[[-1]], axis=1)
            train_x, train_y = realpos_fakepos(real_positive_data3, fake_positive_data2)
            accuracy = knn_model(train_x, train_y)
            print(accuracy)
            outfile.write(str(iteration) + "\t" + str(accuracy)+"\n")
    outfile.close()


