import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from data import read_data
from model import Table
from utils import arg_parse, collate_fn, get_pred, f1_eval


def train():
    args = arg_parse()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    train_features = read_data('./dataset/train.json', tokenizer)
    dev_features = read_data('./dataset/dev.json', tokenizer)

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    dev_dataloader = DataLoader(dev_features, batch_size=args.dev_batch_size, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Table(model, args)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('bert') and not any(nd in n for nd in no_decay)],
         'lr': args.bert_learning_rate, 'weight_decay': args.bert_weight_decay},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and not any(nd in n for nd in no_decay)],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n.startswith('bert') and any(nd in n for nd in no_decay)],
         'lr': args.bert_learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert') and any(nd in n for nd in no_decay)],
         'lr': args.learning_rate, 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = len(train_dataloader) * args.num_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    edi_best, edc_best, eaei_best, eaec_best = -1, -1, -1, -1
    start_epoch = 0
    if os.path.exists(args.path_checkpoint):
        checkpoint = torch.load(args.path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dic"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.num_epoch):
        for step, data in enumerate(train_dataloader):
            model.train()
            input_ids, input_mask, table = data
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            loss, _ = model(input_ids, input_mask, table)
            loss = loss / args.accumulation_steps
            loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dic": optimizer.state_dict(),
                      "epoch": epoch}
        torch.save(checkpoint, args.path_checkpoint)

        results_all, labels_all = [], []
        for data in dev_dataloader:
            model.eval()
            input_ids, input_mask, table = data
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            with torch.no_grad():
                _, results = model(input_ids, input_mask, table)

            for i in range(len(table)):
                results_all.append(get_pred(results[i]))
                labels_all.append(get_pred(table[i]))

        f = f1_eval(results_all, labels_all)
        print(f)

        if f[0] > edi_best:
            edi_best = f[0]
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint, args.edi_checkpoint)
        if f[1] > edc_best:
            edc_best = f[1]
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint, args.edc_checkpoint)
        if f[2] > eaei_best:
            eaei_best = f[2]
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint, args.eaei_checkpoint)
        if f[3] > eaec_best:
            eaec_best = f[3]
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint, args.eaec_checkpoint)


if __name__ == '__main__':
    train()
