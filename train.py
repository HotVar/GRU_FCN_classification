import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader

from GRU_FCN import *
from utils import *

class Exercise_dataset(Dataset):
    def __init__(self, df, label, aug_funcs, aug_rate, device):
        self.df = df
        self.label = label
        self.aug_funcs = aug_funcs
        self.aug_rate = aug_rate
        self.device = device
        self.unique_ids = self.df['id'].unique()
        self.unique_labels = self.label['label'].unique()

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):
        idx = idx % len(self.unique_ids)
        id_ = self.unique_ids[idx]

        sample = dict()
        seq = self.df[self.df['id'] == id_].iloc[:, 2:].values
        seq = self.augmentation_1d(seq)
        sample['seq'] = torch.Tensor(seq)
        sample['label'] = torch.Tensor([self.label[self.label['id'] == id_].iloc[0, 1]])

        return sample

    def augmentation_1d(self, seq):
        for func in self.aug_funcs:
            if np.random.random() < self.aug_rate:
                seq = func(seq)

        # seq = DA_Jitter(seq)
        # seq = DA_Scaling(seq)
        # seq = DA_MagWarp(seq)
        # seq = DA_TimeWarp(seq)
        # # seq = DA_Rotation(seq)
        # # seq = DA_Permutation(seq)
        # # seq = DA_RandSampling(seq)
        return seq


def train():
    data = pd.read_csv('open/train_features.csv')
    label = pd.read_csv('open/train_labels.csv')

    CV = False
    BATCH_SIZE = 128
    N_EPOCHS = 200
    VIS_FREQ = 10
    N_CLASSES = len(label['label'].unique())

    AUG_RATE = 1.0
    aug_funcs = []
    aug_funcs.append(DA_Jitter)
    aug_funcs.append(DA_Scaling)
    aug_funcs.append(DA_MagWarp)
    # aug_funcs.append(DA_TimeWarp)
    # aug_funcs.append(DA_Rotation)
    # aug_funcs.append(DA_Permutation)
    # aug_funcs.append(DA_RandSampling)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    unique_ids = data['id'].unique()

    if CV:
        cv = KFold(n_splits=4,
                   random_state=42,
                   shuffle=True)
        data_yield = enumerate(cv.split(unique_ids))
    else:
        train_data, valid_data = train_test_split(unique_ids,
                                                  test_size=0.25,
                                                  random_state=42,
                                                  shuffle=True)
        data_yield = enumerate(zip([train_data], [valid_data]))

    for k, (train_id, valid_id) in data_yield:
        print(f'Fold {k} ' + '=' * 50)
        X_train = data[data['id'].isin(train_id)]
        y_train = label[label['id'].isin(train_id)]
        X_valid = data[data['id'].isin(valid_id)]
        y_valid = label[label['id'].isin(valid_id)]

        train_dataset = Exercise_dataset(X_train, y_train, aug_funcs, AUG_RATE, device)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_dataset = Exercise_dataset(X_valid, y_valid, aug_funcs, AUG_RATE, device)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        GRU_model = Vanilla_GRU(input_size=6,
                                hidden_size=6,
                                num_layers=1,
                                output_size=len(label['label'].unique()),
                                batch_size=BATCH_SIZE,
                                dropout_rate=0.8,
                                device=device).to(device)
        FCN_model = FCN_1D(in_channels=6,
                           out_channels=128).to(device)
        model = GRU_FCN(GRU=GRU_model,
                        FCN=FCN_model,
                        batch_size=BATCH_SIZE,
                        seq_len=600,
                        n_class=len(label['label'].unique())).to(device)

        loss_CE = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-03, weight_decay=1e-02)

        loss_train = AverageMeter('loss_train', ':.6f')
        loss_valid = AverageMeter('loss_valid', ':.6f')

        for epoch in range(N_EPOCHS):
            for i, (train_sample, valid_sample) in enumerate(zip(train_dataloader, valid_dataloader)):

                train_seq = train_sample['seq'].to(device)
                train_label = train_sample['label'].to(device).long()

                if train_label.size(0) > 1: # multi batch case
                    train_label = train_label.squeeze()
                else: # single batch case
                    train_label = train_label.squeeze(0)

                optim.zero_grad()

                pred = model(train_seq)

                if pred.size(0) == N_CLASSES:
                    pred = pred.unsqueeze(0)

                loss = loss_CE(pred, train_label)
                loss_train.update(loss.item())

                loss.backward()
                optim.step()

                with torch.no_grad():
                    valid_seq = valid_sample['seq'].to(device)
                    valid_label = valid_sample['label'].to(device).long()

                    if valid_label.size(0) > 1:  # multi batch case
                        valid_label = valid_label.squeeze()
                    else:  # single batch case
                        valid_label = valid_label.squeeze(0)

                    pred = model(valid_seq)

                    if pred.size(0) == N_CLASSES:
                        pred = pred.unsqueeze(0)

                    loss = loss_CE(pred, valid_label)
                    loss_valid.update(loss.item())

            if epoch % VIS_FREQ == 0:
                print(f'Epoch[{epoch+1}] : {loss_train}, {loss_valid}')

        torch.save(model, f'{str(datetime.today().date())}_{N_EPOCHS}.pth')


if __name__ == '__main__':
    train()