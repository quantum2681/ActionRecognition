params = dict()

params['num_classes'] = 3

params['dataset'] = 'dataset/'

params['epoch_num'] = 5
params['batch_size'] = 8

params['step'] = 10
params['num_workers'] = 2
params['learning_rate'] = 1e-3
params['betas'] = (0.9, 0.99)
params['weight_decay'] = 1e-5
params['display'] = 2
params['validate'] = 4
params['checkpoint'] = 'checkpoint/checkpoint_0.pt'
# params['checkpoint'] = None
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'checkpoint'
params['clip_len'] = 16
params['frame_sample_rate'] = 1