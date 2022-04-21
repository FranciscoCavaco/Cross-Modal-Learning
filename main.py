import numpy as np

# ? Generate random state to ensure replicable results
np.random.seed(16)

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils import dataUtils, modelUtils, lossFuncs
import yaml


# ? train to be used with ray tune
def train(config, checkpoint_dir=None):
    """
    training:
    --------
    model, algorithm : {epochs, batch_size, criterion, optimizer}
    """
    training_config = config["training"]

    config_data = config["data"]
    conf_splits = [config_data["splits"][split] for split in config_data["splits"]]

    text_datasets, vocabulary = dataUtils.load_data(config_data)

    text_loaders = {}

    for split in conf_splits:
        _dataset = text_datasets[split]  # Extract information about dataset
        # ? batch_size = 32
        _loader = DataLoader(
            dataset=_dataset,
            batch_size=training_config["algorithm"]["batch_size"],
            shuffle=True,
            collate_fn=dataUtils.collate_fn,
        )
        text_loaders[split] = _loader  # ? seperate by the split
    model_config = config[training_config["model"]]  # ? get current model

    model = modelUtils.get_model(model_config, vocabulary)
    print(model)

    # ? get loss and optimiser
    alg_config = training_config["algorithm"]

    lossFunc_config = config[alg_config["criterion"]]
    lossFunc = getattr(lossFuncs, lossFunc_config["name"], None)(**lossFunc_config["args"])

    optimizer_config = config[alg_config["optimizer"]]
    optimizer = getattr(optim, optimizer_config["name"], None)(
        model.parameters(), **optimizer_config["args"]
    )

    lr_scheduler = getattr(optim.lr_scheduler, "ReduceLROnPlateau")(optimizer, **optimizer_config["scheduler_args"])


    #? load checkpoint if it exists
    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

   # modelUtils.train(model, optimizer, lossFunc, text_loaders["development"])


if __name__ == "__main__":
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    train(conf)
