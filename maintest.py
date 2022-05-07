import numpy as np
import ray

from models.pann_pretrain import Transfer_Cnn14

# ? Generate random state to ensure replicable results
np.random.seed(16)

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils import dataUtils, modelUtils, lossFuncs
import yaml
from ray import tune
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
import time
import h5py

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


    #transfer = Transfer_Cnn14(300, True)
    #transfer.load_from_pretrain('./pretrain/Cnn14_16k.pth')

    _loader = DataLoader(
            dataset=text_datasets[conf_splits[0]],
            batch_size=training_config["algorithm"]["batch_size"],
            shuffle=True,
            collate_fn=dataUtils.collate_fn,
        )
    
    model_config = config[training_config["model"]]
    model = modelUtils.get_model(model_config, vocabulary)
    for batch in _loader:
        audio_feats, audio_lens, queries, query_lens, infos = batch
        audio_emb, query_emb = model(audio_feats,queries,query_lens )
        print(query_emb.shape)
        print(audio_emb.shape)
        break
    '''
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
    lossFunc = getattr(lossFuncs, lossFunc_config["name"], None)(
        **lossFunc_config["args"]
    )

    optimizer_config = config[alg_config["optimizer"]]
    optimizer = getattr(optim, optimizer_config["name"], None)(
        model.parameters(), **optimizer_config["args"]
    )

    """
    Reduce learning rate when a metric has stopped improving. 
    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
    """
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **optimizer_config["scheduler_args"]
    )

    # ? load checkpoint if it exists
    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(alg_config["epochs"] + 1):
        if epoch > 0:
            # ? load with training dataset
            modelUtils.train(model, optimizer, lossFunc, text_loaders["development"])

        epoch_results = {}
        for split in conf_splits:
            epoch_results[f"{split}_loss"] = modelUtils.eval(
                model, lossFunc, text_loaders[split]
            )

        # ? reduce lr as validaion_loss stops decreasing
        lr_scheduler.step(epoch_results[config["ray_conf"]["stopper_args"]["metric"]])

        # ? creating checkpoints with ray https://docs.ray.io/en/latest/tune/api_docs/trainable.html#:~:text=score%22%2C%20mode%3D%22max%22))-,Function%20API%20Checkpointing,-%C2%B6
        # ? local_dir/exp_name/trial_name/checkpoint_<step>
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            # ? A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(**epoch_results)  # ? sends the scores to tune
        '''


# modelUtils.train(model, optimizer, lossFunc, text_loaders["development"])


if __name__ == "__main__":
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    train(conf)

    '''
    config_data = conf["data"]
    conf_splits = [config_data["splits"][split] for split in config_data["splits"]]

    ray_conf = conf["ray_conf"]

    # Initialize a Ray cluster
    ray.init(**ray_conf["init_args"])

    """
    They can also stop the entire experiment after a condition is met. 
    For instance, stopping mechanisms can specify to stop trials when 
    they reached a plateau and the metric doesn’t change anymore.
    https://docs.ray.io/en/latest/tune/api_docs/stoppers.html
    """

    # Initialize a trial stopper
    """
    A trial is a set of epochs with a set of val_loss values,
    if the std of these values is lower than the threshold the trial 
    will be stoppped stopped early. This is because the loss isn't chaninging much.  
    """
    stopper = getattr(tune.stopper, ray_conf["trial_stopper"], TrialPlateauStopper)(
        **ray_conf["stopper_args"]
    )

    # Initialize a progress reporter
    """
    This creates a command line interface reporter and reports the results of the model.
    JupyterNotebookReporter -> used for notebooks
    """
    reporter = getattr(tune.progress_reporter, ray_conf["reporter"], CLIReporter)()
    for _split in conf_splits:
        reporter.add_metric_column(metric=f"{_split}_loss")

    #? Function to set the trial name
    def trial_name_creator(trial):
        training_model = conf["training"]["model"]
        trial_name = f"{training_model}_{trial.trial_id}"
        return trial_name

    def trial_dirname_creator(trial):
        trial_dirname = "{0}_{1}_{2}".format(
            conf["training"]["model"], trial.trial_id, time.strftime("%Y-%m-%d_%H-%M-%S")
        )
        return trial_dirname

    
    # Run a Ray cluster - local_dir/exp_name/trial_name
    analysis = tune.run(
        run_or_experiment=train,
        metric=ray_conf["stopper_args"]["metric"], #? validation_loss
        mode=ray_conf["stopper_args"]["mode"],
        name=conf["experiment"],
        stop=stopper,
        config=conf,
        resources_per_trial={
            "cpu": 1,
            "gpu": ray_conf["init_args"]["num_gpus"] / ray_conf["init_args"]["num_cpus"]
        },
        num_samples=1,
        local_dir=conf["output_path"], #? directory to save training results
        # search_alg=search_alg,
        # scheduler=scheduler,
        keep_checkpoints_num=None,
        checkpoint_score_attr=None,
        progress_reporter=reporter,
        log_to_file=True,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        # max_failures=1,
        fail_fast=False,
        # restore="",  # Only makes sense to set if running 1 trial.
        # resume="ERRORED_ONLY",
        reuse_actors=True, #? Tune uses Ray actors to parallelize the evaluation of multiple hyperparameter configurations. 
        raise_on_failed_trial=True
    )

    # Check the best trial and its best checkpoint
    #? Compares all trials’ scores on metric. If metric is not specified, self.default_metric will be used.
    #? https://docs.ray.io/en/latest/tune/api_docs/analysis.html
    best_trial = analysis.get_best_trial(
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        scope="all"  #? look at all scores
    )

    #? checkpoint with best trial
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"]
    )

    print("Best trial:", best_trial.trial_id)
    print("Best checkpoint:", best_checkpoint)'''
