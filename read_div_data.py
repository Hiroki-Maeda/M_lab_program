import numpy as np
import pandas as pd
import os

Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","div_data")

data = pd.read_csv(os.path.join(Data_Path,"data_a Cz-REF.csv"))

print(data.head())

print(data.shape)



