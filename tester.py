import torch
import csv
import json
import os
from sklearn.metrics import classification_report
from tqdm import tqdm

class Tester:
    def __init__(self, args, Simi_encoder, Classifier, test_set, test_size, dataset):
        self.args = args
        self.Simi_encoder = Simi_encoder
        self.classifier = Classifier
        self.test_set = test_set
        self.test_size = test_size
        self.dataset = dataset
        
    def test(self):
        self.Simi_encoder = torch.load(os.path.join(self.args.test_path,f"{self.args.exname}ckpt.Aligner"))
        self.classifier = torch.load(os.path.join(self.args.test_path ,f"{self.args.exname}ckpt.classifier"))
        self.Simi_encoder.eval()
        self.classifier.eval()
        preds = []
        ans_list = []
        id_list = []
        with tqdm(self.test_set) as pbar:
            for batches in pbar:
                
                #######Load Data#########
                post=batches["post"].to(self.args.device)

                img_batch = batches["image"].to(self.args.device)
                dct_img = batches["dct_img"].to(self.args.device)
                ans = batches["label_list"]
                Id = batches["Id"]
                

                
                for ans_label in ans:
                    ans_label = int(ans_label)
                    ans_list.append(ans_label)
                
                for index in Id:
                    id_list.append(index)
                    
                with torch.no_grad():
                    
                    post_align, image_align, simi_matrix, _ = self.Simi_encoder(post, img_batch)
                    output = self.classifier(post, img_batch, post_align, image_align, dct_img, simi_matrix)
                    
                    _, pred= torch.max(output,1)
                    
                    
                for y in pred.cpu().numpy():
                        preds.append(y)
            
        print(classification_report(ans_list, preds, digits=4))
            
        with open(os.path.join(self.args.output_dir, f"{self.args.exname}report.txt"), mode="w") as f:
            f.write(classification_report(ans_list,preds,digits=4))
            
        with open(os.path.join(self.args.output_dir, f"{self.args.exname}result.txt"), mode="w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'label'])
            for i,p in enumerate(preds):
                writer.writerow([id_list[i],p])