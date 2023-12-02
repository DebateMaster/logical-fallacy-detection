from typing import Tuple
import random
import torch
import network.utils.predict
from network.utils.predict import predict_outcome

"""
A mock function simulating the behaviour of the text analyser.
Text anyliser is expected to predict the substring of the text that contains the logical 
fallacy and its specific type.
"""
def predict(text: str) -> str: 
    return predict_outcome(text)
