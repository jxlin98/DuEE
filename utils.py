import argparse
import ujson as json

import torch


def trigger_schema_process():
    def label_add(labels, _type):
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    labels.append(u"O")
    with open("dataset/event_schema.json") as f:
        for line in f:
            d_json = json.loads(line.strip())
            labels = label_add(labels, d_json["event_type"])

    value = range(len(labels))
    return dict(zip(labels, value))


def role_schema_process():
    def label_add(labels, _type):
        if u"{}".format(_type) not in labels:
            labels.extend([u"{}".format(_type)])
        return labels

    labels = []
    labels.append(u"O")
    with open("dataset/event_schema.json") as f:
        for line in f:
            d_json = json.loads(line.strip())
            for role in d_json["role_list"]:
                labels = label_add(labels, role["role"])

    value = range(len(labels))
    return dict(zip(labels, value))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=104, type=int)

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--dev_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)

    parser.add_argument("--event_size", default=131, type=int)
    parser.add_argument("--role_size", default=122, type=int)

    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--logits_dropout", default=0.1, type=float)

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float)
    parser.add_argument("--bert_weight_decay", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_epoch", default=50, type=int)
    parser.add_argument("--path_checkpoint", default="./checkpoint/model.pkl", type=str)
    parser.add_argument("--edi_checkpoint", default="./checkpoint/edi_best.pkl", type=str)
    parser.add_argument("--edc_checkpoint", default="./checkpoint/edc_best.pkl", type=str)
    parser.add_argument("--eaei_checkpoint", default="./checkpoint/eaei_best.pkl", type=str)
    parser.add_argument("--eaec_checkpoint", default="./checkpoint/eaec_best.pkl", type=str)
    args = parser.parse_args()

    return args


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    table = [f["table"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    output = (input_ids, input_mask, table)
    return output


def get_pred(table):
    n = table.shape[0]

    i = 0
    ed_list = []
    while i < n:
        if table[i][i] % 2 == 1:
            num = table[i][i]
            start_pos = i
            while i + 1 < n and table[i+1][i+1] == num + 1:
                i += 1
            end_pos = i
            ed_list.append((start_pos, end_pos, num))
        i += 1

    eae_list = []
    for i in range(len(ed_list)):
        start1, end1, num = ed_list[i]

        r = []
        for ii in range(start1):
            l = []
            for jj in range(start1, end1 + 1):
                l.append(table[ii][jj])
            if len(set(l)) == 1 and l[0] != 0:
                r.append(l[0])
            else:
                r.append(0)

        for ii in range(start1, end1 + 1):
            r.append(0)

        for ii in range(end1 + 1, n):
            l = []
            for jj in range(start1, end1 + 1):
                l.append(table[ii][jj])
            if len(set(l)) == 1 and l[0] != 0:
                r.append(l[0])
            else:
                r.append(0)
        
        k = 0
        while k < n:
            if r[k] != 0:
                num2 = r[k]
                start2 = k
                while k + 1 < n and r[k+1] == num2:
                    k += 1
                end2 = k
                eae_list.append((num, start2, end2, num2))
            k += 1

    return ed_list, eae_list


def f1_eval(results_all, labels_all):
    edid_correct, ed_correct, ed_predict, ed_label = 0, 0, 0, 0
    eaeid_correct, eae_correct, eae_predict, eae_label = 0, 0, 0, 0

    for i in range(len(results_all)):
        result, label = results_all[i], labels_all[i]

        result_ed, label_ed = result[0], label[0]
        for res in result_ed:
            if res in label_ed:
                ed_correct += 1
        ed_predict += len(result_ed)
        ed_label += len(label_ed)

        result_edid, label_edid = [], []
        for j in range(len(result_ed)):
            result_edid.append((result_ed[j][0], result_ed[j][1]))
        for j in range(len(label_ed)):
            label_edid.append((label_ed[j][0], label_ed[j][1]))
        for res in result_edid:
            if res in label_edid:
                edid_correct += 1

        result_eae, label_eae = result[1], label[1]
        for res in result_eae:
            if res in label_eae:
                eae_correct += 1
        eae_predict += len(result_eae)
        eae_label += len(label_eae)

        result_eaeid, label_eaeid = [], []
        for j in range(len(result_eae)):
            result_eaeid.append((result_eae[j][0], result_eae[j][1], result_eae[j][2]))
        for j in range(len(label_eae)):
            label_eaeid.append((label_eae[j][0], label_eae[j][1], label_eae[j][2]))
        for res in result_eaeid:
            if res in label_eaeid:
                eaeid_correct += 1

    f = [0.0] * 4
    p = [edid_correct / ed_predict if ed_predict != 0 else 1, ed_correct / ed_predict if ed_predict != 0 else 1,
         eaeid_correct / eae_predict if eae_predict != 0 else 1, eae_correct / eae_predict if eae_predict != 0 else 1]
    r = [edid_correct / ed_label, ed_correct / ed_label, eaeid_correct / eae_label, eae_correct / eae_label]
    for i in range(4):
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if p[i] + r[i] != 0 else 0

    return f


if __name__ == '__main__':
    labels1 = trigger_schema_process()
    labels2 = role_schema_process()
    print(labels1)
    print(labels2)
