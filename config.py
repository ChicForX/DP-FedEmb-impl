config_dict = {
    "sensitivity": 1.0,
    "noise_multiplier": 0.5,
    "clip_bound_batch": 1.0,
    "batch_size": 128,
    "epochs": 30,
    "pretrain_epochs": 30,
    "num_clients": 5,
    "lr_fedavg": 1e-2,
    "lr_centralized": 1e-2,
    "lr_fedemb_backbone": 1e-2,
    "lr_fedemb_client_backbone": 1e-3,
    "lr_fedemb_client_head": 1e-2,
    "pretrain_lr": 1e-3,
    "pretrained_model_file": "pretrained_model.pt",
    'delta': 1e-5,
    'sample_prop': 1e-3,
    'total_iterations': 468
}