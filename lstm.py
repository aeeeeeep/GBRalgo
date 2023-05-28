import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from find import find
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import *

# run = wandb.init(project="mcm")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
mm_x = StandardScaler()
mm_y = StandardScaler()


class Args:
    def __init__(self) -> None:
        self.batch_size = 256
        self.lr = 1e-3
        self.epochs = 100
        self.num_workers = 12
        self.double_micro = True
        self.full_list = False

        self.in_dim = 8
        self.seq_len = 4
        self.num_layers = 6
        self.hidden_size = 128
        self.radio = 0.8

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = Args()


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.seq_len = args.seq_len
        self.lstm = nn.LSTM(args.in_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_dim, args.hidden_size)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(args.hidden_size, 128)
        self.norm2 = nn.LayerNorm(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, x, test=False):
        h_0 = torch.randn(args.num_layers, x.size(0), args.hidden_size).to('cuda')
        c_0 = torch.randn(args.num_layers, x.size(0), args.hidden_size).to('cuda')
        x, _ = self.lstm(x, (h_0, c_0))
        if not test:
            x = self.dropout1(x)
        x = self.relu1(self.norm1(x))
        x = self.fc2(x)
        if not test:
            x = self.dropout2(x)
        x = self.relu2(self.norm2(x))
        if not test:
            x = self.dropout3(x)
        x = self.fc3(x)
        return x


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    if args.full_list:
        sublist_1 = full_list
    else:
        sublist_1 = full_list[:offset, :]
    sublist_2 = full_list[offset:, :]
    return sublist_1, sublist_2


class Dataset(Dataset):
    def __init__(self, df, flag='train') -> None:
        df.iloc[:, 1:] = df.iloc[:, 1:].clip(lower=0)
        df1 = df.loc[:, ['time', '1_PM2.5', '1_PM10', '1_NO2', '1_temperature', '1_humidity', 'CO mg/m³', 'NO₂ μg/m³',
                         'PM₁₀ μg/m³', 'PM₂.₅ μg/m³', 'SO₂ μg/m³', 'O₃ μg/m³', 'temperature', 'humidity',
                         'pressure hPa', 'wind velocity m/s', 'wind direction']]
        df2 = df.loc[:, ['time', '2_PM2.5', '2_PM10', '2_NO2', '2_temperature', '2_humidity', 'CO mg/m³', 'NO₂ μg/m³',
                         'PM₁₀ μg/m³', 'PM₂.₅ μg/m³', 'SO₂ μg/m³', 'O₃ μg/m³', 'temperature', 'humidity',
                         'pressure hPa', 'wind velocity m/s', 'wind direction']]
        df2.rename(
            columns={'2_PM2.5': '1_PM2.5', '2_PM10': '1_PM10', '2_NO2': '1_NO2', '2_temperature': '1_temperature',
                     '2_humidity': '1_humidity'}, inplace=True)
        if args.double_micro:
            # df1 = df1.append(df2)
            df1 = pd.concat((df1, df2), axis=0)
        # df = hour2sincos(df1)
        # df = df.set_index('time').between_time('21:00','6:00')
        # df = df.set_index('time').between_time('7:00','20:00')
        data = df1.loc[:,
               ['1_PM2.5', '1_PM10', 'PM₁₀ μg/m³', '1_temperature', '1_humidity', 'pressure hPa', 'wind velocity m/s',
                'wind direction']].values
        data = mm_x.fit_transform(data)
        label = multiscale_analysis(df['PM₂.₅ μg/m³'])
        label = mm_y.fit_transform(label.reshape(-1, 1))
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'
        self.label = label
        self.seq_len = args.seq_len
        train_data, val_data = data_split(data, ratio=args.radio, shuffle=False)
        if self.flag == 'train':
            self.data = torch.tensor(train_data, dtype=torch.float32)
            self.len = len(train_data)
        else:
            self.data = torch.tensor(val_data, dtype=torch.float32)
            self.len = len(val_data)

    def __getitem__(self, index):
        if index + self.seq_len >= self.len:
            index = self.len - self.seq_len - 1
        x = torch.as_tensor(self.data[index:index + self.seq_len], dtype=torch.float32)
        y = torch.as_tensor(self.label[index:index + self.seq_len], dtype=torch.float32)
        return y, x

    def __len__(self) -> int:
        return self.len


def train():
    data = pd.read_excel('./resource/features.xlsx')
    print(data.head())
    # data[:,:-2] = train_data
    train_dataset = Dataset(df=data, flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=False)
    val_dataset = Dataset(df=data, flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False, drop_last=True)

    model = Net(args.in_dim, 1).to(args.device)
    # run.watch(model)
    # criterion = nn.HuberLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # , eps=1e-8)

    train_epochs_loss = []
    valid_epochs_loss = []
    RMSEs = []
    MAEs = []

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        # =========================train=======================
        for idx, (label, inputs) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.cpu().item())
        train_epochs_loss.append(np.average(train_epoch_loss))
        print('train loss = {}'.format(np.average(train_epoch_loss)))

        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            pred = []
            labels = []

            for idx, (label, inputs) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = model(inputs, test=True)
                loss = criterion(outputs, label)
                if epoch == args.epochs - 1:
                    outputs = outputs.cpu().squeeze()
                    label = label.cpu()
                    outputs = mm_y.inverse_transform(outputs[:, 0].reshape(-1, 1))
                    label = mm_y.inverse_transform(label[:, 0, :].reshape(-1, 1))
                    for k in range(args.batch_size):
                        pred.append(outputs[k])
                        labels.append(label[k])
                    RMSE = np.sqrt(metrics.mean_squared_error(outputs, label))
                    MAE = metrics.mean_absolute_error(outputs, label)
                    RMSEs.append(RMSE.item())
                    MAEs.append(MAE.item())
                val_epoch_loss.append(loss.item())
            valid_epochs_loss.append(np.average(val_epoch_loss))
            print("epoch = {}, loss = {}".format(epoch, np.average(val_epoch_loss)))
            if epoch == args.epochs - 1:
                plt.plot(labels, label="labels")
                plt.plot(pred, label="pred")
                plt.legend()
                plt.show()
        # run.log({"train_epochs_loss": train_epochs_loss,
        #          "valid_epochs_loss": valid_epochs_loss,
        #          "epoch": epoch
        #          })

    # =========================plot==========================
    # plt.figure(figsize=(12, 4))
    # plt.subplot(121)
    # plt.plot(train_epochs_loss[:])
    # plt.title("train_loss")
    # plt.subplot(122)
    # plt.plot(train_epochs_loss, '-o', label="train_loss")
    # plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    # plt.title("epochs_loss")
    # plt.legend()
    # plt.show()
    # plt.grid()
    # plt.plot(range(len(RMSEs)), MAEs, label='MAE', color='deepskyblue')
    # plt.plot(range(len(RMSEs)), RMSEs, label='RMSE', color='lightpink')
    # plt.rcParams.update({"font.size": 15})
    # plt.title('model evaluation')
    # plt.legend()
    # plt.show()
    # =========================save model=====================
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()
