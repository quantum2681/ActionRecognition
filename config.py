params = dict()

params['num_classes'] = 10

params['dataset'] = 'dataset/'

params['epoch_num'] = 5
params['batch_size'] = 8
params['step'] = 10
params['num_workers'] = 2
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 1
params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'checkpoint'
params['clip_len'] = 32
params['frame_sample_rate'] = 1