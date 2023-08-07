import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class PostsDataset(Dataset):
    def __init__(
        self,
        data,
        mode = "train"
        ):
        super(PostsDataset, self).__init__()
        self.data = data
        self.mode=mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instance = self.data[index]
        label = instance["label"]
        Id = instance["Id"]
        post = instance["post"]
        img = instance["image"]
        dct_img = instance["dct_img"]
        if (self.mode=='test'):
            return  (label, post, img, dct_img, Id, index)
        else:
            return  (label, post, img, dct_img, Id, None)
    
        
    def collate_fn(self, samples):   
        label_list = [int(s[0]) for s in samples]
        label_list=torch.tensor(label_list)
        batch = {"label_list": label_list}
        # ========== news ============== 
        ids=[s[1] for s in samples]
        batch['post']=torch.Tensor(np.array([item.numpy() for item in ids]))
        # ========== Image ============== 
        ids=[s[2] for s in samples]
        batch['image'] = torch.Tensor(np.array([item.numpy() for item in ids]))
        ids=[s[3] for s in samples]
        batch['dct_img'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        # ========== ID ============== 
        Id = [s[4] for s in samples]
        batch['Id'] = Id
        return batch
    

            
            
        
        
        

    