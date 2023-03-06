import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# generate PC6 table
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
        # table[key] = torch.from_numpy(c[0:6, index])
        table[key] = list(c[0:6, index])
    table['X'] = [0,0,0,0,0,0]

    return table

def padding_seq(original_seq, length=50, pad_value='X'):
    padded_seq = original_seq.ljust(length, pad_value)
    return padded_seq

def seq_to_features(seq):
    table = amino_encode_table_6()
    features_list = []
    for aa in seq:
        features_list.append(table[aa])
    feature_tensor = torch.Tensor(features_list)
    return feature_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# """
input_size = 6
sequence_length = 50
num_layers = 2

hidden_size = 4
num_epochs = 20000
batch_size = 256
bidirectional = False
learning_rate = 1e-5
early_stop = 500
weight_decay = 0.02

validate_ratio = 0.9

seed=1403567815
same_seed(seed)

model_path = './model.ckpt'
# """

class HemolysisDataset(Dataset):
    def __init__(self, parquet):
        df = pd.read_parquet(parquet)
        # df['len'] = df['sequence'].str.len()
        # max_seq_len = df['len'].max()   # max_len = 133
        # df = df.drop(['len'], axis=1)
        df['sequence'] = df['sequence'].apply(padding_seq)
        df['encoded'] = df['sequence'].apply(seq_to_features)
        self.features = torch.stack(df['encoded'].values.tolist(), dim=0)
        self.conc = torch.from_numpy(df['concentration'].values)
        self.lysis = torch.from_numpy(df['lysis'].values)
        self.n_samples = df.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.conc[index], self.lysis[index]

    def __len__(self):
        return self.n_samples
        
        
dataset = HemolysisDataset('./train.parquet')
test = HemolysisDataset('./test.parquet')
# TODO: K-Fold Cross Validation
train_subset, validate_subset = random_split(dataset=dataset, lengths=[1-validate_ratio, validate_ratio],generator=torch.Generator().manual_seed(seed))
TrainLoader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
ValidateLoader = DataLoader(dataset=validate_subset, batch_size=batch_size, shuffle=False, num_workers=0)
TestLoader = DataLoader(dataset=test, batch_size=10000000,shuffle=False, num_workers=0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi = 1 if bidirectional else 0
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=0.4)
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight'):
                nn.init.xavier_normal_(param)
        self.act = nn.LeakyReLU(0.2)
        self.do = nn.Dropout(0.4)
        self.bn = nn.BatchNorm1d(hidden_size*(1+self.bi))
        self.fc = nn.Linear(hidden_size*(1+self.bi)+1, 1)
    
    def forward(self, encoded, conc):
        h0 = torch.zeros(self.num_layers*(1+self.bi), encoded.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(encoded, h0)
        out = out[:, -1, :] # extract last time step
        conc = torch.unsqueeze(conc, 1)
        out = self.act(out)
        out = self.do(out)
        out = self.bn(out)
        fc_in = torch.cat((out, conc), dim=1)
        ret = self.fc(fc_in)
        return ret

model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate*100, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 0, -1, False)

# Training loop
n_total_steps = len(TrainLoader)
training_losses = []
validation_losses = []
curr_best_loss = math.inf
for epoch in range(num_epochs):
    total_loss = 0
    total_steps = 0
    model.train()
    for i, (rnn_input, conc, labels) in enumerate(TrainLoader):
        rnn_input = rnn_input.to(device)
        conc = conc.to(device)
        labels = labels.to(device)

        outputs = model(rnn_input, conc)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)
        total_loss  = total_loss + loss.item()
        total_steps = total_steps + 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step(epoch + i / n_total_steps)

    total_val_loss = 0
    total_val_steps = 0
    model.eval()
    with torch.no_grad():
        for i, (rnn_input, conc, labels) in enumerate(ValidateLoader):
            rnn_input = rnn_input.to(device)
            conc = conc.to(device)
            labels = labels.to(device)
            # labels = torch.unsqueeze(labels, dim=1)

            outputs = model(rnn_input, conc)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_val_loss  = total_val_loss + loss.item()
            total_val_steps = total_val_steps + 1

    mean_valid_loss = total_val_loss / total_val_steps

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {total_loss/total_steps:.4f}, Average Validation Loss: {total_val_loss/total_val_steps:.4f}, lr: {scheduler.get_last_lr()[0]}')
    training_losses.append(total_loss/total_steps)
    validation_losses.append(total_val_loss/total_val_steps)
    if (epoch+1) % 500 == 0:
        plt.plot(validation_losses, '-r', label='val')
        plt.plot(training_losses, '-b', label='train')
        plt.xlabel('n epoch')
        plt.savefig(f'epoch{epoch+1}.png')
    
    if mean_valid_loss < curr_best_loss:
        curr_best_loss = mean_valid_loss
        torch.save(model.state_dict(), model_path) # Save your best model
        print('Saving model with loss {:.4f}...'.format(curr_best_loss))
        early_stop_count = 0
    else: 
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print('\nModel is not improving, halt the training session.')
        plt.plot(validation_losses, '-r', label='val')
        plt.plot(training_losses, '-b', label='train')
        plt.xlabel('n epoch')
        plt.savefig(f'final.png')
        break

    # scheduler.step()
    # print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/total_steps:.4f}')

model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional).to(device)
model.load_state_dict(torch.load(model_path))

model.eval()
with torch.no_grad():
    for i, (rnn_input, conc, labels) in enumerate(TestLoader):
        rnn_input = rnn_input.to(device)
        conc = conc.to(device)
        labels = labels.to(device)
        # labels = torch.unsqueeze(labels, dim=1)

        outputs = model(rnn_input, conc)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)
        print(f'test loss:{loss}')