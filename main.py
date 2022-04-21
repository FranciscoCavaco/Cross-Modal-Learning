import numpy as np

# ? Generate random state to ensure replicable results
np.random.seed(16)

from torch.utils.data import DataLoader
from Utils import dataUtils


#? train to be used with ray tune
def train(config, checkpoint_dir=None):
    """
    training:
    --------
    model, algorithm : {epochs, batch_size, criterion, optimizer}
    """
    training_config = config['training']

    config_data = config['data']
    conf_splits = [config_data["splits"][split] for split in config_data["splits"]]

    text_datasets, vocabulary= dataUtils.load_data(config["training"])

    text_loaders = {}

    for split in conf_splits:
        _dataset = text_datasets[split] #Extract information about dataset
        #? batch_size = 32
        _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                             shuffle=True, collate_fn=dataUtils.collate_fn)
    

