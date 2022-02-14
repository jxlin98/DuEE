import torch
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer

from data import read_data
from model import Table
from utils import arg_parse, collate_fn, get_pred, f1_eval


def test():
    args = arg_parse()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    test_features = read_data('./dataset/dev.json', tokenizer)
    test_dataloader = DataLoader(test_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                                  drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Table(model, args)
    model.to(device)

    edi_checkpoint = torch.load(args.edi_checkpoint)
    model.load_state_dict(edi_checkpoint["model_state_dict"])
    results_all, labels_all = [], []
    for data in test_dataloader:
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

    edc_checkpoint = torch.load(args.edc_checkpoint)
    model.load_state_dict(edc_checkpoint["model_state_dict"])
    results_all, labels_all = [], []
    for data in test_dataloader:
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

    eaei_checkpoint = torch.load(args.eaei_checkpoint)
    model.load_state_dict(eaei_checkpoint["model_state_dict"])
    results_all, labels_all = [], []
    for data in test_dataloader:
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

    eaec_checkpoint = torch.load(args.eaec_checkpoint)
    model.load_state_dict(eaec_checkpoint["model_state_dict"])
    results_all, labels_all = [], []
    for data in test_dataloader:
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


if __name__ == '__main__':
    test()
