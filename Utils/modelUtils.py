import copy
from models import core
import torch
import os
# ? Vocabulary is the list of all the embeddings
"""
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
"""


def get_model(config, vocabulary):

    model_args = copy.deepcopy(config["args"])

    # ? This allows us to configure different models
    if config["name"] in ["CRNNWordModel"] or config["name"] in ["PANNWordModel"]:
        # ? this sets the value by pass by value
        embed_args = model_args["text_encoder"]["word_embedding"]
        embed_args["num_word"] = len(vocabulary)
        embed_args["word_embeds"] = (
            vocabulary.get_weights() if embed_args["pretrained"] else None
        )

    # ? also works for pannpretrained
    return getattr(core, config["name"], None)(**model_args)


def train(model, optimizer, loss_fun, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun.to(device=device)
    model.to(device=device)

    """
    https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    model.train() tells your model that you are training the model. 
    So effectively layers like dropout, batchnorm etc. which behave 
    different on the train and test procedures know what is going on 
    and hence can behave accordingly.
    """
    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
        #! Remember: the queries are indexes from the vocabulary
        audio_feats, audio_lens, queries, query_lens, infos = data
        audio_feats, queries = audio_feats.to(device), queries.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

        loss = loss_fun(audio_embeds, query_embeds, infos)
        loss.backward()
        optimizer.step()


def eval(model, loss_fun, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            audio_feats, audio_lens, queries, query_lens, infos = data
            audio_feats, queries = audio_feats.to(device), queries.to(device)

            audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

            loss = loss_fun(audio_embeds, query_embeds, infos)
            eval_loss += loss.cpu().numpy()
            eval_steps += 1

    return eval_loss / (eval_steps + 1e-20)  # ? find the average llss

#? Restore pretrained model
def restore(model, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"), map_location=device)
    print("Setting the evaluation weights")
    
    model.load_state_dict(model_state)
    return model
