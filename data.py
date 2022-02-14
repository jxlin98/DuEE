import numpy as np
import ujson as json

from transformers import BertTokenizer
from utils import trigger_schema_process, role_schema_process


def read_data(file, tokenizer):
    data = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    trigger_schema = trigger_schema_process()
    role_schema = role_schema_process()

    features = []
    for sample in data:
        sentence = []
        for word in sample['text']:
            sentence.append(word)
        input_ids = tokenizer.convert_tokens_to_ids(sentence)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        table = np.zeros([len(input_ids) - 2, len(input_ids) - 2], dtype=int)

        event_list = sample['event_list']
        for event in event_list:
            event_type = event['event_type']
            event_start_pos = event['trigger_start_index']
            table[event_start_pos][event_start_pos] = trigger_schema[f'B-{event_type}']
            for pos in range(event_start_pos + 1, event_start_pos + len(event['trigger'])):
                table[pos][pos] = trigger_schema[f'I-{event_type}']

            for argument in event['arguments']:
                argument_role = argument['role']
                arg_start_pos = argument['argument_start_index']
                for pos_i in range(event_start_pos, event_start_pos + len(event['trigger'])):
                    for pos_j in range(arg_start_pos, arg_start_pos + len(argument['argument'])):
                        table[pos_i][pos_j] = role_schema[f'{argument_role}']
                        table[pos_j][pos_i] = role_schema[f'{argument_role}']

        feature = {'input_ids': input_ids, 'table': table}
        features.append(feature)
    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_features = read_data('dataset/train.json', tokenizer)

    print(train_features[0]['table'])
