params = dict()

params['num_classes'] = 10

params['dataset'] = 'dataset/'

params['epoch_num'] = 200
params['batch_size'] = 10

params['step'] = 200
params['num_workers'] = 2
params['learning_rate'] = 1e-3
params['betas'] = (0.9, 0.99)
params['weight_decay'] = 1e-4
params['display'] = 5
params['validate'] = 5
params['checkpoint'] = 'ActionRecognition/checkpoint/checkpoint_0_.pt'
params['model_path'] = 'ActionRecognition/checkpoint/model.pt'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'checkpoint'
params['clip_len'] = 32
params['frame_sample_rate'] = 1