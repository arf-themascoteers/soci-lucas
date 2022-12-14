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

        soc = inputs[:, 1]
        soc = soc.reshape(-1, 1)

        s471 = (inputs[:,1])
        blue478 = (inputs[:,2])
        s500 = (inputs[:,3])
        s530 = (inputs[:,4])
        green546 = (inputs[:,5])
        b3_560 = (inputs[:,6])
        s590 = (inputs[:,7])
        s620 = (inputs[:,8])
        red659 = (inputs[:,9])
        b4_665 = (inputs[:,10])
        s709 = (inputs[:,11])
        b8_842_nir = (inputs[:,12])
        s844 = (inputs[:,13])
        s1001 = (inputs[:,14])
        s1064 = (inputs[:,15])
        s1094 = (inputs[:,16])
        s1104 = (inputs[:,17])
        s1114 = (inputs[:,18])
        s1185 = (inputs[:,19])
        s1245 = (inputs[:,20])
        s1316 = (inputs[:,21])
        s1498 = (inputs[:,22])
        s1558 = (inputs[:,23])
        s1588 = (inputs[:,24])
        s1790 = (inputs[:,25])
        s1982 = (inputs[:,26])
        s1992 = (inputs[:,27])
        s2052 = (inputs[:,28])
        s2102 = (inputs[:,29])
        s2103 = (inputs[:,30])
        s2150 = (inputs[:,31])
        s2163 = (inputs[:,32])
        s2355 = (inputs[:,33])

        soci = blue478/(red659 * green546)
        soci = soci.reshape(-1,1)
        soci_14 = soci**1.4
        inv_soci = 1 / soci

        ndvi = (b8_842_nir-red659)/(b8_842_nir+red659)
        ndvi = ndvi.reshape(-1,1)

        # ndvi = ndvi.reshape(-1,1)
        # sn = ndvi / soci
        # inv_soci = 1/soci
        # soci1_4 = soci**1.4
        # soci_1_by_4 = soci**(1/4)

        # si_1001_471 = (s1001 - s471)/(s1001 + s471)
        # si_1001_471 = si_1001_471.reshape(-1,1)

        # si_1001_500 = (s1001 - s500)/(s1001 + s500)
        # si_1001_500 = si_1001_500.reshape(-1,1)

        # si_1001_2150 = (s1001 - s2150)/(s1001 + s2150)
        # si_1001_2150 = si_1001_2150.reshape(-1,1)

        # bi = (((red659*red659) + (green546*green546))**(1/2)) /2
        # bi = bi.reshape(-1,1)

        # bi2 = (((red659*red659) + (green546*green546)+ (b8_842_nir*b8_842_nir))**(1/2)) /3
        # bi2 = bi2.reshape(-1,1)

        # my_bi = (((red659*red659) + (green546*green546) + (blue478*blue478))**(1/2)) /2
        # my_bi = my_bi.reshape(-1,1)
        #
        # my_bi2 = red659 + green546 + blue478
        # my_bi2 = my_bi2.reshape(-1,1)
        #
        # my_bi3 = (red659 + green546 + blue478)**(1/2)
        # my_bi3 = my_bi3.reshape(-1,1)
        #
        # my_bi4 = ((red659*red659) + (green546*green546) + (blue478*blue478))**(1/3)
        # my_bi4 = my_bi4.reshape(-1,1)

        # my_bi5 = (-(red659*red659) + (green546*green546) + (blue478*blue478))**(1/3)
        # my_bi5 = my_bi5.reshape(-1,1)

        # my_bi6 = (-(red659*red659) + (green546*green546) + (blue478*blue478))**(1/2)
        # my_bi6 = my_bi6.reshape(-1,1)

        # my_bi7 = ( (green546*green546*green546) + (blue478*blue478*blue478) - (red659*red659*red659))**(1/3)
        # my_bi7 = my_bi7.reshape(-1,1)

        # my_bi8 = ( (green546*green546*green546*green546) + (blue478*blue478*blue478*blue478) - (red659*red659*red659*red659))**(1/4)
        # my_bi8 = my_bi8.reshape(-1,1)

        # my_bi9 = ( (green546**5) + (blue478**5) - (red659**5))**(1/5)
        # my_bi9 = my_bi9.reshape(-1,1)

        # my_bi10 = ( (green546**6) + (blue478**6) - (red659**5))**(1/6)
        # my_bi10 = my_bi10.reshape(-1,1)

        # my_bi11 = ( (green546**7) + (blue478**7) - (red659**5))**(1/7)
        # my_bi11 = my_bi11.reshape(-1,1)

        # my_bi12 = ( (green546**8) + (blue478**8) - (red659**8))**(1/8)
        # my_bi12 = my_bi12.reshape(-1,1)
        #
        # my_bi13 = ( (green546**9) + (blue478**9) - (red659**9))**(1/9)
        # my_bi13 = my_bi13.reshape(-1,1)

        # my_bi14 = ( (green546**20) + (blue478**20) - (red659**20))**(1/20)
        # my_bi14 = my_bi14.reshape(-1,1)
        #
        my_bi15 = ( (green546**40) + (blue478**40) - (red659**40))**(1/40)
        my_bi15 = my_bi15.reshape(-1,1)

        my_bi16 = ( (green546**50) + (blue478**80) - (red659**80))**(1/80)
        my_bi16 = my_bi16.reshape(-1,1)

        self.x = my_bi15
        self.y = soc

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

