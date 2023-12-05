import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true") 
parser.add_argument("--num_prototypes", type=int, default=40)
parser.add_argument("--num_pos_prototypes", type=int, default=40)
parser.add_argument("--model", type=str, default="ProtoTEx")
parser.add_argument("--modelname", type = str)
parser.add_argument("--data_dir", type = str)
parser.add_argument("--model_checkpoint", type=str, default=None)

# Wandb parameters
parser.add_argument("--project", type=str)
parser.add_argument("--experiment", type=str)
parser.add_argument("--none_class", type=str, default="No")
parser.add_argument("--nli_intialization", type=str, default="Yes")
parser.add_argument("--curriculum", type=str, default="No")
parser.add_argument("--architecture", type=str, default="BART")
parser.add_argument("--augmentation", type=str, default="No")


args = parser.parse_args()

datasets_config =  {
    "../data/logical_fallacy_with_none": {
        'features': {
            'text': 'source_article', 
            'label': 'updated_label'
        },
        'classes': {
            'O': 0,
            'ad hominem': 1,
            'ad populum': 2,
            'appeal to emotion': 3,
            'circular reasoning': 4,
            'fallacy of credibility': 5,
            'fallacy of extension': 6,
            'fallacy of logic': 7,
            'fallacy of relevance': 8,
            'false causality': 9,
            'false dilemma': 10,
            'faulty generalization': 11,
            'intentional': 12,
            'equivocation': 13
        }
    },
    "../data/bigbench": {
        'features': {
            'text': 'text',
            'label': 'label'
        }, 
        'classes': {
            0: 0,
            1: 1
        }
    },
}

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]
