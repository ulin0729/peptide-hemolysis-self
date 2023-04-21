# pickle file did the import part implicitly. However, for the sake of making the dependencies clearer, I still import the modules explicitly here.
import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
from Bio import SeqIO
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb  
import argparse

# load precalculated pc6 table
try:
    with open('pc6.pickle', 'rb') as f:
        pc6_table = pickle.load(f)
except IOError:
    print("can't open pc6.pickle")
    sys.exit(1)

def padding_seq(sequences_dict, length=50, pad_value='X'):
    ret = {}
    for key,_ in sequences_dict.items():
        ret[key] = {'seq':sequences_dict[key]['seq'].ljust(length, pad_value).upper(), 'conc':sequences_dict[key]['conc']}
    return ret

def PC6_encoding(data):
    ret = {}
    conc = {}
    for key in data.keys():
        integer_encoded = []
        div = data[key]['conc']/300
        for amino in data[key]['seq']:
            integer_encoded += (pc6_table[amino]+[div])
        ret[key]=integer_encoded
        conc[key]=data[key]['conc']
    return ret,conc

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Peptide sequence hemolysis predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta name',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    parser.add_argument('-t','--threshold',help='the threshold of the classifier',required=True,type=int)
    args = parser.parse_args()
    input_fasta_name = args.fasta_name
    output_csv_name =  args.output_csv
    threshold = args.threshold
    
    # invalid input handling
    if threshold % 10 != 0:
        print('Threshold is not a multiple of 10.')
        sys.exit(1)
    if threshold > 50:
        print('Warning! Classifiers with threshold over 50 are trained on insufficient dataset. The result might be biased.')
    try:
        fasta_f = open(input_fasta_name, 'r')
        fasta_sequences = SeqIO.parse(fasta_f,'fasta')
    except IOError:
        print(f"can't open {input_fasta_name}")
        sys.exit(1)
    try:
        with open(f'{threshold}_clf.pickle', 'rb') as f:
            clf = pickle.load(f)
    except IOError:
        print(f"can't open {threshold}_clf.pickle")
        sys.exit(1)

    # encode the input sequences
    sequences_dict = {}
    for fasta in fasta_sequences:
        new_row = dict.fromkeys(['seq', 'conc'])
        name, new_row['seq'] = str(fasta.id), str(fasta.seq)
        descriptions = name.split('|')
        ID = descriptions[0]
        for col in descriptions[1:]:
            if col.startswith('Conc='):
                try:
                    new_row['conc'] = np.float32(col[5:])
                    if new_row['conc'] <= 0 or new_row['conc'] > 300:
                        new_row['conc'] = np.float32(50)
                    else:
                        break
                except ValueError:
                    new_row['conc'] = np.float32(50)
        else:
            new_row['conc'] = np.float32(50)
        if len(new_row['seq']) > 50:
            # print(f'Length of {ID} is greater than 50, cropping it into first 50 amino acids.')
            new_row['seq'] = new_row['seq'][:50]
        sequences_dict[ID] = new_row
    
    padded_dict = padding_seq(sequences_dict)
    encoded_dict, conc_dict = PC6_encoding(padded_dict)
    df = pd.DataFrame(encoded_dict).T
    conc_df = pd.DataFrame(conc_dict, index=[0]).T
    prediction = clf.predict(df.values) # actual prediction
    df['pred'] = prediction
    df['conc'] = conc_df
    df = df.drop(range(0,350), axis=1)
    df.to_csv(args.output_csv, header=False)
    