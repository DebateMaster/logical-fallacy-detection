from typing import Tuple
import random
import torch
import network.utils.predict
from network.utils.predict import predict_outcome

def predict(text: str) -> Tuple[bool, str]: 
    return predict_outcome(text)
