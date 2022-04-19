def train(config, checkpoint_dir=None ):
    '''
    model, algorithm : {epochs, batch_size, criterion, optimizer}
    '''
    training_config = config['training']