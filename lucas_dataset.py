import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import math


class LucasDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.csv_file_location = "data/lucas-many.csv"
        self.scaler = None
        df = pd.read_csv(self.csv_file_location)
        train, test = model_selection.train_test_split(df, test_size=0.2)
        #df = self._preprocess(df)
        inputs = torch.tensor(df[df.columns[0:]].values, dtype=torch.float32)

        soc = inputs[:, 0].reshape(-1, 1)

        s471 = self.transform(inputs[:,1])
        blue478 = self.transform(inputs[:,2])
        s500 = self.transform(inputs[:,3])
        s530 = self.transform(inputs[:,4])
        green546 = self.transform(inputs[:,5])
        b3_560 = self.transform(inputs[:,6])
        s590 = self.transform(inputs[:,7])
        s620 = self.transform(inputs[:,8])
        red659 = self.transform(inputs[:,9])
        b4_665 = self.transform(inputs[:,10])
        s709 = self.transform(inputs[:,11])
        b8_842_nir = self.transform(inputs[:,12])
        s844 = self.transform(inputs[:,13])
        s1001 = self.transform(inputs[:,14])
        s1064 = self.transform(inputs[:,15])
        s1094 = self.transform(inputs[:,16])
        s1104 = self.transform(inputs[:,17])
        s1114 = self.transform(inputs[:,18])
        s1185 = self.transform(inputs[:,19])
        s1245 = self.transform(inputs[:,20])
        s1316 = self.transform(inputs[:,21])
        s1498 = self.transform(inputs[:,22])
        s1558 = self.transform(inputs[:,23])
        s1588 = self.transform(inputs[:,24])
        s1790 = self.transform(inputs[:,25])
        s1982 = self.transform(inputs[:,26])
        s1992 = self.transform(inputs[:,27])
        s2052 = self.transform(inputs[:,28])
        s2102 = self.transform(inputs[:,29])
        s2103 = self.transform(inputs[:,30])
        s2150 = self.transform(inputs[:,31])
        s2163 = self.transform(inputs[:,32])
        s2355 = self.transform(inputs[:,33])


        ndvi = (b8_842_nir-red659)/(b8_842_nir+red659)

        bi = (((red659*red659) + (green546*green546))**(1/2)) /2

        bi2 = (((red659*red659) + (green546*green546)+ (b8_842_nir*b8_842_nir))**(1/2)) /3

        my_bi = (((red659*red659) + (green546*green546) + (blue478*blue478))**(1/2)) /2

        si_1 = 1/(blue478) #0.46
        si_2 = 1/(red659) #0.43
        si_3 = 1/(green546) #0.498

        si1 = 1/(blue478**2) #0.5
        si2 = 1/(red659**2) #0.39
        si3 = 1/(green546**2) #0.499

        soci = blue478/(red659 * green546) #0.43

        si4 = 1/(blue478**2)

        self.x = si4
        self.y = soc

    def transform(self, array):
        return (1/10**array).reshape(-1,1)

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

