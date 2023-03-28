import sklearn
import pickle
import math
import numpy as np
import pandas as pd

def amino_encode_table_6(): # key: Amino Acid, value: tensor
    df = pd.read_csv('./6-pc', sep=' ', index_col=0)
    H1 = (df['H1'] - np.mean(df['H1'])) / (np.std(df['H1'], ddof=1))
    V = (df['V'] - np.mean(df['V'])) / (np.std(df['V'], ddof=1))
    P1 = (df['P1'] - np.mean(df['P1'])) / (np.std(df['P1'], ddof=1))
    Pl = (df['Pl'] - np.mean(df['Pl'])) / (np.std(df['Pl'], ddof=1))
    PKa = (df['PKa'] - np.mean(df['PKa'])) / (np.std(df['PKa'], ddof=1))
    NCI = (df['NCI'] - np.mean(df['NCI'])) / (np.std(df['NCI'], ddof=1))
    c = np.array([H1,V,P1,Pl,PKa,NCI], dtype=np.float32)
    amino = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    table = {}
    for index,key in enumerate(amino):
        table[key] = list(c[0:6, index])
    table['X'] = [0,0,0,0,0,0]
    return table

table = amino_encode_table_6()

def padding_seq(original_seq: str, length=50, pad_value='X')->str:
    padded_seq = original_seq.ljust(length, pad_value)
    return padded_seq

def seq_to_features_ml(seq:str, conc:float)->np.array:
    features_list = []
    for aa in seq:
        t = table[aa].copy()
        t.append(conc)
        features_list += t
    feature_tensor = np.array(features_list, dtype=np.float32)
    return feature_tensor



def predict(seq:str, conc:float, threshold:int) -> list:
    feature = seq_to_features_ml(padding_seq(seq),conc)
    feature = np.expand_dims(feature, axis=0)   # shape of feature should be [n_samples, 350]
    if threshold == 0:  # use all combined
        clfs = [pickle.load(open(f'{i}_clf.pickle', 'rb')) for i in range(10,100,10)]
        pred_res = []
        for i,clf in enumerate(clfs, 1):
            pred = clf.predict(feature).item(0)
            pred_res.append(pred)
            # print(f'predicted by {i*10}_clf: {"hemolysis" if pred else "not hemolysis"}')
        return pred_res
    else:
        clf = pickle.load(open(f'{threshold}_clf.pickle', 'rb'))
        prediction = clf.predict(feature)
        # print(f'predicted by {threshold}_clf: {"hemolysis" if prediction.item(0) else "not hemolysis"}')
        return prediction

def post_process(prediction: list):
    correct_prob = [0.8,0.8,0.8,0.8,0.8,0.7,0.8,0.6,0.7]
    prob = [1]*(len(prediction)+1)
    for i in range(9):
        if prediction[i] == 0:
            prob[0] *= correct_prob[i]
        else:
            prob[0] *= 1-correct_prob[i]
    for i in range(1,10):
        for j in range(i):
            if prediction[j] == 1:
                prob[i] *= correct_prob[j]
            else:
                prob[i] *= 1-correct_prob[j]
        for j in range(i,9):
            if prediction[j] == 0:
                prob[i] *= correct_prob[j]
            else:
                prob[i] *= 1-correct_prob[j]
    return prob

def argmax(l:list)->int:
    temp = -math.inf
    ret = 0
    for i,x in enumerate(l):
        if x > temp:
            temp = x
            ret = i
    return ret

if __name__ == '__main__':
    df = pd.read_parquet('test.parquet')
    df['prediction'] = df.apply(lambda x: predict(x['sequence'], x['concentration'], 0),axis=1)
    df['prob'] = df['prediction'].apply(lambda x: post_process(x))
    df['argmax'] = df['prob'].apply(lambda x: argmax(x))
    del df['prob']
    df.to_parquet('./temp.parquet', index=False)
    n = len(df)
    correct = [0] * 10
    mse = 0
    for _,r in df.iterrows():
        mse += (abs(r['lysis']-(r['argmax']*10+5)) ** 2)
        for i in range(10):
            if r['lysis']-(r['argmax']*10) <= (10*i) and r['lysis']-(r['argmax']*10) >= 0:
                correct[i] += 1
    print(df.to_string())
    for i in range(10):
        print(f'allowed error: {i*10}, acc: {correct[i]/n}')
    print(f'mse: {mse/n}')
    
    '''
    allowed error: 0, acc: 0.11486486486486487
    allowed error: 10, acc: 0.6283783783783784
    allowed error: 20, acc: 0.7162162162162162
    allowed error: 30, acc: 0.7736486486486487
    allowed error: 40, acc: 0.8614864864864865
    allowed error: 50, acc: 0.9087837837837838
    allowed error: 60, acc: 0.9425675675675675
    allowed error: 70, acc: 0.9594594594594594
    allowed error: 80, acc: 0.9864864864864865
    allowed error: 90, acc: 0.9932432432432432
    mse: 820.8590743503261
    '''