class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/2024/DMTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/2024/DMTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/2024/DMTrack/pretrained_networks'
        self.got10k_val_dir = '/root/DMTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/root/DMTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/DMTrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/root/DMTrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/root/DMTrack/data/coco_lmdb'
        self.coco_dir = '/root/DMTrack/data/coco'
        self.lasot_dir = '/root/DMTrack/data/lasot'
        self.got10k_dir = '/root/DMTrack/data/got10k/train'
        self.trackingnet_dir = '/root/DMTrack/data/trackingnet'
        self.depthtrack_dir = '/data/datasets/depthtrack/DepthTrackTraining'
        self.lasher_dir = '/data/datasets/LasHeR/TrainingSet'
        self.visevent_dir = '/data/datasets/visevent/train_subset'
