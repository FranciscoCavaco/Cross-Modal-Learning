import copy
from models import core
import torch

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
    if config["name"] in ["CRNNWordModel"]:
        # ? this sets the value by pass by value
        embed_args = model_args["text_encoder"]["word_embedding"]
        embed_args["num_word"] = len(vocabulary)
        embed_args["word_embeds"] = (
            vocabulary.get_weights() if embed_args["pretrained"] else None
        )

        return getattr(core, config["name"], None)(**model_args)

    return None


def train(model, optimizer, loss_fun, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fun.to(device=device)
    model.to(device=device)

    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, infos)
        audio_feats, audio_lens, queries, query_lens, infos = data
        audio_feats, queries = audio_feats.to(device), queries.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

        loss = loss_fun(audio_embeds, query_embeds, infos)
        loss.backward()
        optimizer.step()
