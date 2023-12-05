import torch
import transformers
from transformers import RobertaModel
import numpy as np
from torch import cuda 

device = 'cuda' if cuda.is_available() else 'cpu'

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

