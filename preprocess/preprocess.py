import torch
import pickle
from transformers import BertTokenizer, ViTImageProcessor, ViTModel, BertModel, logging
import sys 
from backbones import VisionTransformer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import PostDataset
from argparse import ArgumentParser, Namespace
from pathlib import Path
import warnings
from fractions import Fraction



def getter(args, dataset):
    #####Extractor######
    Bert_model = BertModel.from_pretrained('').to(args.device)
    visual_enoder = VisionTransformer.VIT(model_dim = args.model_dim).to(args.device)
    vit_dict = visual_enoder.state_dict()
    pre_weights = torch.load(args.visual_pretrained, map_location = args.device)#.state_dict()
    pretrained_dict = {k: v for k, v in pre_weights.items() if k in vit_dict}
    vit_dict.update(pretrained_dict)

    
    
    visual_enoder.load_state_dict(vit_dict)
    for param in Bert_model.parameters():
        param.requires_grad = False 
    for param in visual_enoder.parameters():
        param.requires_grad = False 
        
    ######################
    Final_Dataset = []
    with tqdm(dataset) as Pbar:
        for batches in Pbar:
            input_ids_post=batches["input_ids_post"].to(args.device)
            masks_post=batches["masks_post"].to(args.device)
            dct_img = batches["dct_img"].to(args.device)
            
            
            img_batch = batches["image"].to(args.device)
            
            y = batches["label_list"]
            
            Id = batches["Id"]
            
            post, o2_post = Bert_model(input_ids = input_ids_post, attention_mask = masks_post, return_dict = False)

            
            _, output = visual_enoder(img_batch)
            
            for i in range(len(y)):
                thisdict = {
                    "Id": Id[i],
                    "label": y[i],
                    "post": torch.cat([o2_post.unsqueeze(1), post], dim = 1)[i].cpu(),
                    "image":output[i].cpu(),
                    "dct_img": dct_img[i].cpu()
                }
                Final_Dataset.append(thisdict)
    return Final_Dataset
            
def main(args, getter):
    dataset = pickle.load(open(args.data_dir, 'rb'))
    senmantic_Bert_tokenizer = BertTokenizer.from_pretrained("")
    
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    datasets = PostDataset(dataset, senmantic_Bert_tokenizer, args.max_sent_len, args.max_sent_num,  data_transforms, args.filter)
    final_set = DataLoader(datasets, batch_size = args.batch_size,collate_fn = datasets.collate_fn,
                shuffle = True, drop_last = False,
                num_workers = 0, pin_memory = False)
    with torch.no_grad():
        Final_Data = getter(args, final_set)

    pickle.dump(Final_Data, open(f"{args.dataname}.pkl", "wb"))
    
def fraction_type(value):
    try:
        numerator, denominator = value.split('/')
        result = float(numerator) / float(denominator)
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid fraction value: {}".format(value))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type = Path,
        help = "Directory to the dataset",
        default = "",
    )
    parser.add_argument(
        "--visual_pretrained",
        type=Path,
        help="Directory to load the pretrained visual model.",
        default="",
    )
    
    parser.add_argument("--max_sent_len", type=int, default = 120)
    parser.add_argument("--max_sent_num", type=int, default = 100)
    
    parser.add_argument("--filter", type=fraction_type, default=9/16)
    
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=128)
    
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:5"
    )

    parser.add_argument("--dataname", type=str, default="")
     
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args, getter)
