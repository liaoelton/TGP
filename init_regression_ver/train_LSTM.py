import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
# from transformers import BertTokenizer, BertModel

import torch
from tqdm import trange, tqdm
from dataset import SeqClsDataset
# from utils import Vocab, get_cnt_per_batch
from model import SeqClassifier
from torch.utils.data import DataLoader

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

"""
{
    + : 0,
    - : 1,      
    * : 2,    => word embedding ? (future work)
    / : 3,
}

[ 0 , 1 , 1 ]
"""


def main(args):
    seed = 5
    torch.manual_seed(seed)

    """
    with open(args.cache_dir / "best/vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "best/intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    """
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: create DataLoader for train / dev datasets -> DONE
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }
    #embeddings = torch.load(args.cache_dir / "embeddings.pt").to(args.device)

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        # num_class = datasets[TRAIN].num_classes,
        device = args.device
    )
    #print(model)

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        training_loss = 0
        t = 0
        model.train()
        bar = tqdm(dataloaders[TRAIN])
        for idx, batch in enumerate(bar):
            batch['program'] = batch['program'].to(args.device)
            batch['output'] = batch['output'].to(args.device)
            optimizer.zero_grad()
            res = model(batch)
            # bar.set_postfix(loss=loss.item(), iter=i, lr=optimizer.param_groups[0]['lr'])
            loss = res["loss"]
            #print("loss:", loss)
            loss.backward()
            optimizer.step()

            training_loss += loss
            t+=1
        print("avg training loss:", training_loss/t)
        # TODO: Evaluation loop - calculate accuracy and save model weights
    #     model.eval()
    #     bar = tqdm(dataloaders[DEV])
    #     total_cnt = 0
    #     acc_cnt = 0
    #     for idx, batch in enumerate(bar):
    #         batch['program'] = batch['program'].to(args.device)
    #         batch['output'] = batch['output'].to(args.device)
    #         res = model(batch)  
    #         _acc, _total = get_cnt_per_batch(res["y_pred"].detach().cpu(), batch["output"].detach().cpu())
    #         total_cnt += _total
    #         acc_cnt += _acc

    #     val_acc = float(acc_cnt/total_cnt)
    #     print("epoch: ", epoch, ", validation acc: ", val_acc)

    #     if val_acc > best_acc:
    #         ckp_path = ckpt_dir / '{}-model.pth'.format(epoch + 1)
    #         best_ckp_path = ckpt_dir / 'best-model.pth'.format(epoch + 1)
    #         torch.save(model.state_dict(), ckp_path)
    #         torch.save(model.state_dict(), best_ckp_path)
    #         print('Saved model checkpoints into {}...'.format(ckp_path))
    #         best_acc = val_acc
    # print("best acc: ", best_acc)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../gp_tree_data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/output/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/output/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    
    parser.add_argument("--num_epoch", type=int, default=10000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
