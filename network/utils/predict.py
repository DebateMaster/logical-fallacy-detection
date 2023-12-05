import os
import pandas as pd
import torch
from typing import Tuple
from transformers import ElectraTokenizer, RobertaTokenizer

from network.utils.models import ProtoTEx_electra
from network.utils.roBERTa import RobertaClass

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
electra_tokenizer = ElectraTokenizer.from_pretrained("howey/electra-base-mnli")
fine_grained_model = ProtoTEx_electra(
    num_prototypes=50,
    num_pos_prototypes=49,
    n_classes=14,
    bias=False,
    dropout=False,
    special_classfn=True, 
    p=1, 
    batchnormlp1=True,
).cuda()

binary_model = RobertaClass(2, 'roberta-base').cuda()
binary_model.to("cuda")
binary_model.eval()


def upload_model(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    model.eval()

upload_model(fine_grained_model, '../models/curr_finegrained_nli_electra_prototex')

def tokenize(tokenizer, text):
    text = str(text)
    text = " ".join(text.split())

    inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
        )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
    ids = ids[None, :].long()
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
    mask = mask[None, :].long()
    return ids.to('cuda'), mask.to('cuda')


def binary_tokenize(tokenizer, text):
    text = str(text)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
    ids = ids[None, :].long()
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
    mask = mask[None, :].long()
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
    token_type_ids = token_type_ids[None, :].long()
    return ids.to('cuda'), mask.to('cuda'), token_type_ids.to('cuda')

classes = {
        0: 'O',
        1: 'ad hominem',
        2: 'ad populum',
        3: 'appeal to emotion',
        4: 'circular reasoning',
        5: 'fallacy of credibility',
        6: 'fallacy of extension',
        7: 'fallacy of logic',
        8: 'fallacy of relevance',
        9: 'false causality',
        10: 'false dilemma',
        11: 'faulty generalization',
        12: 'intentional',
        13: 'equivocation'
}

def predict_class(text: str) -> str: 
    classfn_out = fine_grained_model.forward(*tokenize(electra_tokenizer, text), use_decoder=False, use_classfn=1)
    if classfn_out.ndim == 1:
        predict = torch.zeros_like(y)
        predict[classfn_out > 0] = 1
    else:
        predict = torch.argmax(classfn_out, dim=1)
    return classes[predict[0].tolist()]

def predict_fallacy(text: str) -> str:
    classfn_out = binary_model.forward(*binary_tokenize(roberta_tokenizer, text))
    if classfn_out.ndim == 1:
        predict = torch.zeros_like(y)
        predict[classfn_out > 0] = 1
    else:
        predict = torch.argmax(classfn_out, dim=1)
    return predict[0].tolist()


def predict_outcome(text: str) -> Tuple[bool, str]:
    prediction = predict_fallacy(text)
    if prediction == 0:
        return False, 'There is no fallacy.'
    else:
        prediction = predict_class(text)
        if prediction == 'O':
            fallacy = 'There is some fallacy, but we couldn\'t figure what exactly.'
        else:
            fallacy = f'There is {prediction}.'
        return True, fallacy
