import random
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import torch
from transformers import  logging
from typing import Dict
from Dataset import PostsDataset
from Model import InformationDetection
from torch.utils.data import DataLoader
import copy
from train import Trainer
from Align_function import global_align, local_align
from model_encoder import Similarity_space, DCTDetectionModel
from tester import Tester




TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]


def pickle_reader(path):
    return pickle.load(open(path, "rb"))


def main(args):
    
    
    data_paths = {split: args.data_dir / f"{split}.pkl" for split in SPLITS}
    data = {split: pickle_reader(path) for split, path in data_paths.items()}

    datasets : Dict[str, PostsDataset] = {
        split: PostsDataset(split_data, mode = split)
        for split, split_data in data.items()
    }

    for split, split_dataset in datasets.items():
        if split == "train" and args.mode==0:
            tr_size = len(split_dataset)
            print("tr_size:",tr_size)
            tr_set = DataLoader(
                split_dataset,  batch_size = args.batch_size,collate_fn = split_dataset.collate_fn,
                shuffle = True, drop_last = True,
                num_workers = 0, pin_memory = False)
        elif split == "eval" and args.mode == 0:
            dev_size=len(split_dataset)
            print("dev_size:",dev_size)
            dev_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=True,
                num_workers=0, pin_memory=False)
        elif args.mode == 1:
            test_size=len(split_dataset)
            print("test_size:",test_size)
            test_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last = True,
                num_workers=0, pin_memory=False)

    Sim_encoder = Similarity_space(args.shared_dim)
    DCT_encoder = DCTDetectionModel(embedding_dim = args.model_dim)
    classifier = InformationDetection(DCT_encoder, model_dim = args.model_dim)
    Sim_encoder.to(args.device)
    classifier.to(args.device)



    G_a = global_align().to(args.device)
    L_a = local_align().to(args.device)


    ifexist=os.path.exists(args.output_dir)
    if not ifexist:
        os.makedirs(args.output_dir)


    if args.mode==0: #train/dev
        args_dict_tmp = vars(args)
        args_dict = copy.deepcopy(args_dict_tmp)
        with open(os.path.join(args.output_dir, f"param_{args.exname}.txt"), mode="w") as f:
            f.write("============ parameters ============\n")
            print("============ parameters =============")
            for k, v in args_dict.items():
                f.write(f"{k}: {v}\n")
                print(f"{k}: {v}")
        trainer=Trainer(args, classifier, Sim_encoder, DCT_encoder, G_a, L_a, tr_set, tr_size, dev_set, dev_size)
        trainer.train()
    else: #test
        tester=Tester(args, Sim_encoder, classifier, test_set,test_size,datasets["test"])
        tester.test()

        
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type = Path,
        help = "Directory to the dataset",
        default = "dataset/",
    )

    parser.add_argument(
        "--cache_dir",
        type = Path,
        help = "Directory to the preprocessed caches.",
        default = "./cache/",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./output1/",
    )



    parser.add_argument(
        "--test_path",
        type=Path,
        help="Directory to load the test model.",
        default="./ckpt/",
    )

    parser.add_argument("--shared_dim", type=float, default=128)
    parser.add_argument("--model_dim", type=float, default=64)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--clr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--UPGRADE_RATIO", type=int, default=1)
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:2"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--exname", type=str, default="ex0")

    #
    parser.add_argument("--mode", type=int, help="train:0, test:1", default=0)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

    
        
        
