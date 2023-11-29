from typing import Tuple
import random
import torch
from transformers import RobertaModel, RobertaTokenizer

class RobertaClass(torch.nn.Module):
    """
    1. Class associated with the model - RoBERTa Large  
    2. Attributes: 
      * output_params - number - (number of prediction classes) 
      * model_name - string  
    """
    def __init__(self, output_params, model_name):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, output_params)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def tokenize(tokenizer, text):
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
    return ids, mask, token_type_ids

_model = RobertaClass(2, 'roberta-base')
_model.load_state_dict(torch.load('network/models/binary_roberta_sd.pt'))
_model.eval()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

"""
A mock function simulating the behaviour of the text analyser.
Text anyliser is expected to predict the substring of the text that contains the logical 
fallacy and its specific type.
"""
def predict(text: str) -> Tuple[int, int, str]: 
    model_outputs = _model.forward(*tokenize(tokenizer, text))
    _, max_indices = torch.max(model_outputs.data, dim=1)
    result = max_indices[0].tolist()
    if result == 0:
        return 'That\'s a fallacy'
    else:
        return 'That\'s not a fallacy'