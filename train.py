import torch
from torch import nn
import os
import json
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.metrics import classification_report
from torch.nn import functional as F
import numpy as np
import copy
from numpy import mean
torch.autograd.set_detect_anomaly(True)


def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)

    return fixed_text, matched_image




def evaluation(outputs, labels):
    return torch.sum(torch.eq(outputs, labels)).item()


def get_top_k_indices(similarity_matrix, k_0=0.2):
    batch_size, n1, n2 = similarity_matrix.shape

    row_mask = torch.zeros((batch_size, n2), dtype=torch.float32, device=similarity_matrix.device)
    col_mask = torch.zeros((batch_size, n1), dtype=torch.float32, device=similarity_matrix.device)

    _, row_indices_temp = torch.topk(similarity_matrix[:, 0, 1:], k=int(k_0 * (n2 - 1)), dim=-1)
    row_indices_temp += 1
    row_mask[:, row_indices_temp] = 1

    _, col_indices_temp = torch.topk(similarity_matrix[:, 1:, 0], k=int(k_0 * (n1 - 1)), dim=-1)
    col_indices_temp += 1
    col_mask[:, col_indices_temp] = 1

    return row_mask, col_mask


class Trainer:
    def __init__(self, 
                 args,
                 InformationDetection, 
                 Similarity_Space,
                 DCTDetectionModel,
                 GlobalAlign,
                 LocalAlign,
                 tr_set,
                 tr_size, 
                 dev_set, 
                 dev_size):
        self.args = args 
        self.Simi_encoder = Similarity_Space
        self.dct_encoder = DCTDetectionModel
        self.classifier = InformationDetection
        self.tr_set = tr_set
        self.tr_size = tr_size
        self.dev_set = dev_set
        self.dev_size = dev_size
        self.nloss = nn.CrossEntropyLoss()
        self.GlobalAlign = GlobalAlign
        self.assloss = nn.BCELoss()
        #self.GlobalAlign = GlobalAlign
        self.LocalAlign = LocalAlign
    

    
    
    def train(self):
        

    
        NET_Align = optim.Adam(self.Simi_encoder.parameters(),lr=self.args.lr, weight_decay = 1e-6, eps = 1e-4)
        NET_Classifier = optim.Adam(self.classifier.parameters(), lr = self.args.clr,  weight_decay = 1e-6, eps = 1e-4)
        
        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_acc=0
        for epoch in epoch_pbar:
            self.Simi_encoder.train()
            self.classifier.train()
            align_acc = []
            cls_loss = []
            or_cls_loss = []
            ass_losses = []
            G_Align = []
            dis_l = []
            sim_l = []
            L_Align = []
            Align = []
            total_train_acc = []
            CLASS_GRAD = True
            Align_GRAD = True
            with tqdm(self.tr_set) as pbar:
                for batches in pbar:
                    pbar.set_description(desc = f"Epoch{epoch}")
                    #######Load Data#########
                    post=batches["post"].to(self.args.device)
                    y = batches["label_list"].to(self.args.device)
                    #Id = batches["Id"]
                    img_batch = batches["image"].to(self.args.device)
                    dct_img = batches["dct_img"].to(self.args.device)
                    fixed_text, matched_image = prepare_data(post, img_batch, y)

                    
                    ############Frozen Parameters########    
                    if epoch % self.args.UPGRADE_RATIO == 0:
                        Align_GRAD = True
                        for p in self.Simi_encoder.parameters():
                            p.requires_grad = True
                    else:
                        for p in self.Simi_encoder.parameters():
                            p.requires_grad = False
                            Align_GRAD = False
                            
                    
                            
                    ######Alignment########
                    
                    text_aligned, image_aligned, similarity_matrix = self.Simi_encoder(fixed_text, matched_image)
                   

                    L_aglobal = self.GlobalAlign(text_aligned[:, 0], image_aligned[:, 0])
             
                    
                    L_local = self.LocalAlign(similarity_matrix[:, 1:, 1:])
                    
                    loss_align = L_aglobal + L_local
                    

                    if Align_GRAD:
                        NET_Align.zero_grad()
                        loss_align.backward()
                        NET_Align.step()


                    Align.append(loss_align.item())
                    
                    G_Align.append(L_aglobal.item())

                    L_Align.append(L_local.item())

                    #####classifier#####

                    post_align, image_align, simi_matrix= self.Simi_encoder(post, img_batch)
                    pred = self.classifier(post, img_batch, post_align, image_align, dct_img, simi_matrix)
 
                    _, label = torch.max(pred, dim = 1)
                    
                    y = y.to(torch.long)                 
                    class_loss = self.nloss(pred, y) 
                    correct = evaluation(label, y)/len(label)
                    
                    
                    


                    
                    NET_Classifier.zero_grad()
                    class_loss.backward()
                    NET_Classifier.step()
                    
                    
                    
                    cls_loss.append(class_loss.item())
                    total_train_acc.append(correct)
                    pbar.set_postfix(loss = class_loss.item())
                    
                    
                    
                    
            train_align_json = {"epoch": epoch, "Align_loss": mean(Align), "Global_loss": mean(G_Align), "Local_loss":mean(L_Align)}   
            train_info_json = {"epoch": epoch,"Class_loss": mean(cls_loss), "train_Acc":mean(total_train_acc)}    
            if Align_GRAD :
                print(f"{'#' * 3} Align: {str(train_align_json)} {'#'* 3}")
            
            if CLASS_GRAD:
                print(f"{'#' * 10} TRAIN: {str(train_info_json)} {'#' * 10}")
            
            #STEP += 1
            
            with open(os.path.join(self.args.output_dir, f"log{self.args.exname}.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            
            
                
            if CLASS_GRAD:
                self.Simi_encoder.eval()
                self.classifier.eval()  
                valid_acc = []
                ans_list = []
                preds = []
                
                with torch.no_grad():
                    for batches in self.dev_set:

                        #######Load Data#########
                        post=batches["post"].to(self.args.device)
                        y = batches["label_list"].to(self.args.device)
                        #Id = batches["Id"]
                        img_batch = batches["image"].to(self.args.device)
                        dct_img = batches["dct_img"].to(self.args.device)
                        
                        for ans_label in y:
                            ans_label = int(ans_label)
                            ans_list.append(ans_label)
                        
                        y = y.to(self.args.device)
                        post_align, image_align, simi_matrix= self.Simi_encoder(post, img_batch)
                        pred = self.classifier(post, img_batch, post_align, image_align, dct_img, simi_matrix)
                        _, label= torch.max(pred,1)
                        
                        y=y.to(torch.long)
                        correct = evaluation(label, y)/len(label)
                    
                        valid_acc.append(correct)
                        
                        for p in label.cpu().numpy():
                            preds.append(p)
                            
                            
                print(classification_report(ans_list, preds, digits=4))
                with open(os.path.join(self.args.output_dir, f"log{self.args.exname}.txt"), mode="a") as f:
                    f.write(classification_report(ans_list,preds, digits=4))
                
                valid_info_json = {"epoch": epoch,"valid_Acc":mean(valid_acc)}
                print(f"{'#' * 10} VALID: {str(valid_info_json)} {'#' * 10}")
                
                with open(os.path.join(self.args.output_dir, f"log{self.args.exname}.txt"), mode="a") as fout:
                    fout.write(json.dumps(valid_info_json) + "\n")
            
                if mean(valid_acc) > best_acc:
                    best_acc = mean(valid_acc)
                    torch.save(self.Simi_encoder, f"{self.args.ckpt_dir}/{self.args.exname}ckpt.Aligner")
                    torch.save(self.classifier, f"{self.args.ckpt_dir}/{self.args.exname}ckpt.classifier")
                    print('saving model with acc {:.3f}\n'.format(mean(valid_acc)))
                    with open(os.path.join(self.args.output_dir, f"{self.args.exname}best_valid_log.txt"), mode="a") as fout:
                        fout.write(json.dumps(valid_info_json) + "\n")
                
                
                
         
         
 
        
