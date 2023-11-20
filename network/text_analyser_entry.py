from typing import Tuple
import random

"""
A mock function simulating the behaviour of the text analyser.
Text anyliser is expected to predict the substring of the text that contains the logical 
fallacy and its specific type.
"""
def predict(text: str) -> Tuple[int, int, str]: 
    logical_fallacies = ['strawman', 'false_cause', 'slippery_slope'] 
    i = random.randint(0, len(text) - 2)
    j = random.randint(i + 1, len(text) - 1)
    fallacy = random.choice(logical_fallacies)

    return i, j, fallacy