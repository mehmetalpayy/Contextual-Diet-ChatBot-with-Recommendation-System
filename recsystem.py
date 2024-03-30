import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from random import shuffle

def Recommend(age,weight,height):

    data = pd.read_csv('meals.csv')

    bfdata = data['Breakfast'].to_numpy()
    ldata = data['Lunch'].to_numpy()
    ddata = data['Dinner'].to_numpy()
    Food_itemsdata = data['Food_items']

    bfsp = []
    lsp = []
    dsp = []
    bfspid = []
    lspid = []
    dspid = []

    for i in range(len(Food_itemsdata)):
        if bfdata[i] == 1:
            bfsp.append(Food_itemsdata[i])
            bfspid.append(i)
        if ldata[i] == 1:
            lsp.append(Food_itemsdata[i])
            lspid.append(i)
        if ddata[i] == 1:
            dsp.append(Food_itemsdata[i])
            dspid.append(i)

    lspid_data = data.iloc[lspid]
    lspid_data = lspid_data.T
    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    lspid_data = lspid_data.iloc[Valapnd]
    lspid_data = lspid_data.T

    bfspid_data = data.iloc[bfspid]
    bfspid_data = bfspid_data.T
    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    bfspid_data = bfspid_data.iloc[Valapnd]
    bfspid_data = bfspid_data.T

    dspid_data = data.iloc[dspid]
    dspid_data = dspid_data.T
    val = list(np.arange(5, 15))
    Valapnd = [0] + val
    dspid_data = dspid_data.iloc[Valapnd]
    dspid_data = dspid_data.T

    bmi = weight / ((height / 100) ** 2)

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp + 20)
        for i in test_list:
            if (i == age):
                agecl = round(lp / 20)

    Dinner_ID_numpy = dspid_data.to_numpy()
    Lunch_ID_numpy = lspid_data.to_numpy()
    BF_ID_numpy = bfspid_data.to_numpy()
    ti = (bmi + agecl) / 2

    if (bmi < 16):
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        clbmi = 1
    elif (bmi >= 30):
        clbmi = 0

    Datacalorie = Dinner_ID_numpy[1:, 1:len(Dinner_ID_numpy)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
    dnrlbl = kmeans.labels_

    Datacalorie = Lunch_ID_numpy[1:, 1:len(Lunch_ID_numpy)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
    lnchlbl = kmeans.labels_

    Datacalorie = BF_ID_numpy[1:, 1:len(BF_ID_numpy)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
    brklbl = kmeans.labels_

    data_nut = pd.read_csv('nutrients.csv')
    data_nut.head(5)

    dataTog = data_nut.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    wl = dataTog.iloc[[1, 2, 7, 8]]
    wl = wl.T
    wg = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    wg = wg.T
    h = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    h = h.T
    wl_data = wl.to_numpy()
    wg_data = wg.to_numpy()
    h_data = h.to_numpy()
    wl = wl_data[1:, 0:len(wl_data)]
    wg = wg_data[1:, 0:len(wg_data)]
    h = h_data[1:, 0:len(h_data)]

    wl_nut = np.zeros((len(wl) * 5, 6), dtype=np.float32)
    wg_nut = np.zeros((len(wg) * 5, 10), dtype=np.float32)
    h_nut = np.zeros((len(h) * 5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []

    for zz in range(5):
        for jj in range(len(wl)):
            valloc = list(wl[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            wl_nut[t] = np.array(valloc)
            if jj < len(brklbl):
                yt.append(brklbl[jj])
            t += 1
        for jj in range(len(wg)):
            valloc = list(wg[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            wg_nut[r] = np.array(valloc)
            if jj < len(lnchlbl):
                yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(h)):
            valloc = list(h[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            h_nut[s] = np.array(valloc)
            if jj < len(dnrlbl):
                ys.append(dnrlbl[jj])
            s += 1

    clf = RandomForestClassifier(n_estimators=100)

    if bmi<25 and bmi>18.5:
        X_test = np.zeros((len(h) * 5, 9), dtype=np.float32)

        for i in range(len(h)):
            valloc = list(h[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test[i] = np.array(valloc) * ti

        X_train = h_nut[:len(ys)]
        y_train = ys

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        rec_list = []

        X_test2 = np.zeros((len(wg) * 5, 10), dtype=np.float32)

        for i in range(len(wg)):
            valloc = list(wg[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test2[i] = np.array(valloc) * ti

        X_train2 = wg_nut[:len(yr)]
        y_train2 = yr

        clf.fit(X_train2, y_train2)
        y_pred2 = clf.predict(X_test2)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                try:
                    rec_list.append(Food_itemsdata[i])
                except:
                    pass

        for i in range(len(y_pred2)):
            if y_pred2[i] == 0:
                try:
                    a_list = []
                    a_list.append(Food_itemsdata[i])
                    a = a_list[0]
                    if a not in rec_list:
                        rec_list.append(Food_itemsdata[i])
                except:
                    pass

        shuffle(rec_list)
        rec_list = rec_list[:len(rec_list) // 3]

        string = ""
        for i in range(len(rec_list)):
            string += rec_list[i]+"\n"

        return string

    elif bmi>25:
        X_test2 = np.zeros((len(h) * 5, 9), dtype=np.float32)

        for i in range(len(h)):
            valloc = list(h[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test2[i] = np.array(valloc) * ti

        X_train2 = h_nut[:len(ys)]
        y_train2 = ys

        clf.fit(X_train2, y_train2)
        y_pred2 = clf.predict(X_test2)
        rec_list = []

        X_test = np.zeros((len(wl) * 5, 6), dtype=np.float32)

        for i in range(len(wl)):
            valloc = list(wl[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test[i] = np.array(valloc) * ti

        X_train = wl_nut[:len(yt)]
        y_train = yt

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                try:
                    rec_list.append(Food_itemsdata[i])
                except:
                    pass

        for i in range(len(y_pred2)):
            if y_pred2[i] == 0:
                try:
                    a_list = []
                    a_list.append(Food_itemsdata[i])
                    a = a_list[0]
                    if a not in rec_list:
                        rec_list.append(Food_itemsdata[i])
                except:
                    pass

        shuffle(rec_list)
        rec_list = rec_list[:len(rec_list) // 3]

        string = ""
        for i in range(len(rec_list)):
            string += rec_list[i]+"\n"

        return string

    elif bmi<18.5:
        X_test2 = np.zeros((len(h) * 5, 9), dtype=np.float32)

        for i in range(len(h)):
            valloc = list(h[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test2[i] = np.array(valloc) * ti

        X_train2 = h_nut[:len(ys)]
        y_train2 = ys

        clf.fit(X_train2, y_train2)
        y_pred2 = clf.predict(X_test2)
        rec_list = []

        X_test = np.zeros((len(wg) * 5, 10), dtype=np.float32)

        for i in range(len(wg)):
            valloc = list(wg[i])
            valloc.append(agecl)
            valloc.append(clbmi)
            X_test[i] = np.array(valloc) * ti

        X_train = wg_nut[:len(yr)]
        y_train = yr

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                try:
                    rec_list.append(Food_itemsdata[i])
                except:
                    pass

        for i in range(len(y_pred2)):
            if y_pred2[i] == 0:
                try:
                    a_list = []
                    a_list.append(Food_itemsdata[i])
                    a = a_list[0]
                    if a not in rec_list:
                        rec_list.append(Food_itemsdata[i])
                except:
                    pass

        shuffle(rec_list)
        rec_list = rec_list[:len(rec_list)//3]

        string = ""
        for i in range(len(rec_list)):
            string += rec_list[i]+"\n"

        return string
