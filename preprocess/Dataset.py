import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from nltk import tokenize
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import numpy as np
from scipy.fftpack import fft,dct
from torch.nn import functional as F
import math

        


class PostDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer_bert, 
        max_sent_len: int,
        max_sent_num: int,
        Transforms, 
        filter,
        mode = "train"  
        ):
        super(PostDataset, self).__init__()
        self.tokenizer_bert = tokenizer_bert
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.data = data
        self.mode=mode
        self.transforms = Transforms
        self.filter = filter
        
    def __len__(self):
        return len(self.data)
    
    
    @staticmethod
    def zigzag(data):
        (r, c) = data.shape
        if(r != c):
            print("ERROR")
            return 0
        ZZ = np.zeros((1, r*c))
        p = 0
        for index in range(2 * r):
            if(index <= r-1):
                for i in range(index+1):
                    R = i
                    C = index - i
                    ZZ[0, p] = data[R, C]
                    p = p + 1
            if(index > r-1):
                for i in range(2*r - index-1):
                    C = r-1 - i
                    R = index - C
                    ZZ[0, p] = data[R, C]
                    p += 1
        return ZZ

   
    def process_dct_img(self, img):
        img = img.numpy()
        channel = img.shape[0]
        height = img.shape[1]
        width = img.shape[2]
        N = 7
        step = int(height/N) 

        dct_img = np.zeros((channel, N**2, step*step), dtype=np.float32)



        
        for cn in range(channel):
            i = 0
            for row in np.arange(0, height, step):
                for col in np.arange(0, width, step):
                    block = np.array(img[:, row:(row+step), col:(col+step)], dtype=np.float32)
                    #print('block:{}'.format(block.shape))
                    block1 = block.reshape(1, 3, step**2, 1)
                    dct_temp = dct(block1[:, cn]).reshape(step, step)
                    dct_temp = self.zigzag(dct_temp)
                    
                    dct_img[cn, i, :] = dct_temp 

                    i += 1

    
        
        dct_img = torch.from_numpy(dct_img).float()
        
        dct_img = dct_img[:, :, int(dct_img.shape[-1]*(1-self.filter)):]
        dct_img = dct_img.reshape(3, int(math.sqrt(self.filter)*height), int(math.sqrt(self.filter)*width))

        return dct_img
    
    def __getitem__(self, index):
        instance = self.data[index]
        label = instance["label"]

        news_count=0
        
        Id = instance["post_id"]
        newstext=""
        for sent in tokenize.sent_tokenize(instance["post_content"]):
            if(news_count>=self.max_sent_num):
                break
            newstext+=str(sent)
            news_count+=1
            
        inputs = self.tokenizer_bert.encode_plus(
            newstext,
            padding = 'max_length',
            truncation=True,
            dd_special_tokens=True,
            return_attention_mask=True,
        )
        ids_news = torch.tensor(inputs['input_ids'])
        masks_news=torch.tensor(inputs['attention_mask'])
        
        

        ############################img#####################
        
        img = self.transforms(instance["image"])
        ###############
        transform_dct = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
            ])
        dct_img = transform_dct(instance["image"].convert('YCbCr'))
        dct_img = self.process_dct_img(dct_img)

        
        if (self.mode=='test'):
            return  (label,ids_news,masks_news, img, dct_img, Id, index)
        else:
            return  (label,ids_news,masks_news, img, dct_img, Id, None)
    
        
    def collate_fn(self, samples) :   
        batch={}
        label_list=[]
            
        for s in samples:
            label_list.append(int(s[0]))
        label_list=torch.tensor(label_list)
        batch["label_list"]=label_list

        # ========== news ============== 
        ids=[s[1] for s in samples]
        masks=[s[2] for s in samples]
        
        ids=pad_sequence(ids,batch_first=True)
        masks=pad_sequence(masks,batch_first=True)
    
        ids=torch.as_tensor(ids)
        masks=torch.as_tensor(masks)

        batch['input_ids_post']=ids
        batch['masks_post']=masks
        
        # ========== image ============== 
        img =[s[3] for s in samples]
        img = torch.tensor(np.array([item.cpu().detach().numpy() for item in img]))
        batch['image']= img
        #===========dct_img=================
        dct_img =[s[4] for s in samples]
        dct_img = torch.tensor(np.array([item.cpu().detach().numpy() for item in dct_img]))
        batch['dct_img']= dct_img
        # ========== ID ==============   
        Id = [s[5] for s in samples]
        batch['Id'] = Id
            
        return batch
    

            
            
        
        
        

    
