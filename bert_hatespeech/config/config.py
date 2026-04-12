def config():
    return {
        'BERT': {
            'model_name':        'bert-base-uncased',
            'max_seq_length':    64,
            'num_labels':        2,
            'dropout':           0.1,
            'warmup_ratio':      0.06,
            'weight_decay':      0.01,
            'adam_epsilon':      1e-8,
            'use_class_weights': True,
            'stage1_epochs':     3,
            'stage1_batch_size': 32,
            'search': {
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'epochs':        [2, 3, 4],
                'batch_size':    [16, 32],
            },
        },
        'PATHS': {
            'train_path':     './data/train.csv',
            'test_path':      './data/test.csv',
            'annotated_path': './data/annotated.csv',
            'text_col':       'tweets',
            'label_col':      'label',
            'output_dir':     'outputs/bert_hatespeech',
        },
    }