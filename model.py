import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Table(nn.Module):
    def __init__(self, model, args):
        super(Table, self).__init__()
        self.bert = model
        self.config = model.config

        self.bert_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        self.logits_dropout = nn.Dropout(args.logits_dropout)

        self.table = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

        self.out1 = nn.Linear(self.config.hidden_size, args.event_size)
        self.out2 = nn.Linear(self.config.hidden_size, args.role_size)

    def forward(self, input_ids, input_mask, table):
        output = self.bert(input_ids=input_ids, attention_mask=input_mask)
        sequence_output = self.bert_dropout(output[0])

        x, y = torch.broadcast_tensors(sequence_output[:, :, None], sequence_output[:, None])
        s = self.dropout(F.gelu(self.table(torch.cat([x, y], dim=-1))))

        total_loss = 0
        results = []
        for k in range(len(table)):
            n = table[k].shape[0]
            table_embedding = s[k, 1:n + 1, 1:n + 1]
            labels = torch.tensor(table[k]).to(input_ids.device)
            loss_fct = nn.CrossEntropyLoss(reduction='none')

            event_logits = self.out1(table_embedding)
            mask1 = torch.eye(n).int().to(input_ids.device)
            loss1 = loss_fct(self.logits_dropout(event_logits).flatten(0, 1), (labels * mask1).flatten())
            total_loss += torch.masked_select(loss1, mask1.bool().flatten()).mean()

            argument_logits = self.out2(table_embedding)
            mask2 = (torch.ones(n, n) - torch.eye(n)).int().to(input_ids.device)
            loss2 = loss_fct(self.logits_dropout(argument_logits).flatten(0, 1), (labels * mask2).flatten())
            total_loss += torch.masked_select(loss2, mask2.bool().flatten()).mean()

            event_res = torch.argmax(event_logits, dim=2)
            argument_res = torch.argmax(argument_logits + argument_logits.transpose(0, 1), dim=2)
            res = event_res * mask1 + argument_res * mask2
            results.append(res.to('cpu').numpy())

        return total_loss, results
