import copy
from models import core;
#? Vocabulary is the list of all the embeddings
'''
Config:

# Model hyper-parameters
CRNNWordModel:
    name: CRNNWordModel
    args:
        audio_encoder: main
            in_dim: 64
            out_dim: 300
            up_sampling: True
        text_encoder:
            word_embedding:
                embed_dim: 300
                pretrained: True
                trainable: False
''' 
def get_model(config, vocabulary):

    model_args = copy.deepcopy(config['args'])

    #? This allows us to configure different models 
    if config["name"] in ["CRNNWordModel"]:
        #? this sets the value by pass by value 
        embed_args = model_args["text_encoder"]["word_embedding"]
        embed_args["num_word"] = len(vocabulary)
        embed_args["word_embeds"] = vocabulary.get_weights() if embed_args["pretrained"] else None

       
        return getattr(core, config["name"], None)(**model_args)

    print('args:', model_args)
    return None
