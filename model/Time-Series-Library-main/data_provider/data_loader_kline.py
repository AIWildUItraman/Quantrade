# data_provider/data_loader_custom.py  
import pandas as pd  
import numpy as np  
from torch.utils.data import Dataset  
from sklearn.preprocessing import StandardScaler  
import os  
  
class Dataset_Classification_Custom(Dataset):  
    def __init__(self, root_path, flag='train', data_path='your_data.csv', scale=True):  
        self.root_path = root_path  
        self.data_path = data_path  
        self.scale = scale  
        self.flag = flag  
        self.__read_data__()  
      
    def __read_data__(self):  
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))  
          
        # 划分训练集和测试集
        num_train = int(len(df_raw) * 0.95)  
          
        if self.flag == 'TRAIN':  
            df = df_raw[:num_train]  
        else:  # TEST  
            df = df_raw[num_train:]  
          
        # 特征列  
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']  
        self.feature_df = df[feature_cols]  
          
        # 标签  
        self.labels = df['label'].values  
          
        # 标准化  
        if self.scale:  
            self.scaler = StandardScaler()  
            if self.flag == 'TRAIN':  
                self.scaler.fit(self.feature_df.values)  
            self.feature_df = pd.DataFrame(  
                self.scaler.transform(self.feature_df.values),  
                columns=feature_cols  
            )  
          
        # 分类任务需要的属性  
        self.class_names = ['0', '1', '2']  
        self.max_seq_len = len(self.feature_df)  
          
    def __getitem__(self, index):  
        # 返回整个序列和对应标签  
        return (  
            self.feature_df.iloc[index].values.astype(np.float32),  
            self.labels[index]  
        )  
      
    def __len__(self):  
        return len(self.feature_df)