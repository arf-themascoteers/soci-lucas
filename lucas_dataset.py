import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


class LucasDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.csv_file_location = "data/lucas-4.csv"
        self.scaler = None
        df = pd.read_csv(self.csv_file_location)
        train, test = model_selection.train_test_split(df, test_size=0.2)
        df = self._preprocess(df)
        inputs = torch.tensor(df[df.columns[1:5]].values, dtype=torch.float32)
        blue = inputs[:,0]
        green = inputs[:,1]
        red = inputs[:,2]
        nir = inputs[:,3]
        soci = blue/(red * green)
        soci = soci.reshape(-1,1)
        #soci_s = soci * soci
        #soci_sr = torch.sqrt(soci)
        ndvi = (nir-red)/(nir+red)
        ndvi = ndvi.reshape(-1,1)
        sn = ndvi / soci
        inv_soci = 1/soci
        self.x = torch.cat((ndvi, inputs), dim=1)
        self.y = torch.tensor(df[df.columns[0]].values, dtype=torch.float32)

    def _preprocess(self, df):
        df = self.__scale__(df)
        return df

    def __scale__(self, df):
        df, self.scaler  = self.__scale_col__(df, "soc")
        return df

    def __scale_col__(self, df, col):
        x = df[[col]].values.astype(float)
        a_scaler = MinMaxScaler()
        x_scaled = a_scaler.fit_transform(x)
        df[col] = x_scaled
        return df, a_scaler

    def unscale(self, value):
        return self.scaler.inverse_transform([[value]])[0][0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        this_x = self.x[idx]
        soc = self.y[idx]
        return torch.tensor(this_x, dtype=torch.float32),  torch.tensor(soc, dtype=torch.float32)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


if __name__ == "__main__":
    cid = LucasDataset()
    print(cid.unscale(0.5))
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)

