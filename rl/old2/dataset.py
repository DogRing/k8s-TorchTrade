import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np

def angle_encoding(dt_index,weights=None):
    weights = weights or { 'hour':0.1,'day':0.4,'week':0.3,'year':0.2 }
    timestamps = dt_index.view('int64') // 10**9
    hour_angle = (timestamps%3600) / 3600 * 2 * np.pi
    day_angle = (timestamps%86400) / 86400 * 2 * np.pi
    week_angle = (timestamps%604800) / 603800 * 2 * np.pi
    year_seconds = 365.25 * 86400
    year_angle = (timestamps % year_seconds)/year_seconds*2*np.pi
    sin_sum = (
        weights['hour']*np.sin(hour_angle)+
        weights['day']*np.sin(day_angle)+
        weights['week']*np.sin(week_angle)+
        weights['year']*np.sin(year_angle)
    )
    cos_sum = (
        weights['hour']*np.cos(hour_angle)+
        weights['day']*np.cos(day_angle)+
        weights['week']*np.cos(week_angle)+
        weights['year']*np.cos(year_angle)
    )
    final_angle = (np.arctan2(sin_sum,cos_sum)+np.pi)/(2*np.pi)
    return pd.Series(final_angle,index=dt_index,name='time')

class MultiTimeDataset(Dataset):
    def __init__(self,path,tick,input_dims,batch_size=64,device='cpu'):
        super(MultiTimeDataset,self).__init__()
        self.timeframes = input_dims.keys()
        file_name = f'{path}{tick}.csv'
        df=pd.read_csv(file_name,parse_dates=[0],index_col=[0])
        self.x = {k:torch.tensor(
            df[[col for col in df.columns if col.startswith(k)]].values,
            device=device, dtype=torch.float32
        ) for k in self.timeframes}
        non_tf_cols = [col for col in df.columns
                        if not any(col.startswith(prefix) for prefix in self.timeframes)
                        and col != 'close']
        self.x['1'] = torch.tensor(
            df[non_tf_cols].values,
            device=device, dtype=torch.float32
        )
        self.times = torch.tensor(
            angle_encoding(df.index).values,
            device=device, dtype=torch.float32
        )
        time_indices = {
            k:np.arange(-int(k)*(v-1),int(k),int(k)) 
            for k,v in input_dims.items() 
        }
        om = min(arr.min() for arr in time_indices.values())
        non_nan_index = df.index.get_loc(df.index[~df.isna().any(axis=1)][0])
        self.indices = {
            k:arr + abs(om) + non_nan_index
            for k,arr in time_indices.items()
        }
        self.len = len(df)-(abs(om)+non_nan_index)
        self.y = df.close[non_nan_index+abs(om):]
        self._precompute_first_batch_indices(batch_size)
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return {
            tf: (
                self.x[tf][self.indices[tf]+idx],
                self.times[self.indices[tf]+idx]
            )
            for tf in self.timeframes
        },self.y.iloc[idx]
    def _precompute_first_batch_indices(self,batch_size):
        self.batch_size = batch_size
        total_samples = len(self)
        self.num_batches = (total_samples+ self.batch_size - 1) // self.batch_size
        self.first_batch_indices = {}
        first_batch_size = min(self.batch_size,total_samples)
        for tf in self.timeframes:
            tf_indices = self.indices[tf]
            indices_matrix = np.empty((first_batch_size,len(tf_indices)),dtype=np.int32)
            for i in range(first_batch_size):
                indices_matrix[i] = tf_indices + i
            self.first_batch_indices[tf] = indices_matrix
    def _get_batch_indices(self,batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self))
        return start_idx, end_idx
    def _prepare_batch(self, batch_idx):
        if batch_idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx,end_idx = self._get_batch_indices(batch_idx)
        current_batch_size = end_idx - start_idx
        batch_data = {}
        for tf in self.timeframes:
            base_indices = self.first_batch_indices[tf][:current_batch_size]
            adjusted_indices = base_indices + start_idx
            x_batch = self.x[tf][adjusted_indices]
            times_batch = self.times[adjusted_indices]
            batch_data[tf] = (x_batch,times_batch)
        batch_labels = self.y[start_idx:end_idx]
        return batch_data, batch_labels
    def iter_batch(self):
        for batch_idx in range(self.num_batches):
            yield self._prepare_batch_fast(batch_idx)
    def get_batch(self,batch_idx):
        return self._prepare_batch(batch_idx)
