from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/DMTrack/data/got10k_lmdb'
    settings.got10k_path = '/root/DMTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/DMTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/root/DMTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/DMTrack/data/lasot_lmdb'
    settings.lasot_path = '/root/DMTrack/data/lasot'
    settings.network_path = '/data/2024/DMTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/DMTrack/data/nfs'
    settings.otb_path = '/root/DMTrack/data/otb'
    settings.prj_dir = '/data/2024/DMTrack'
    settings.result_plot_path = '/data/2024/DMTrack/output/test/result_plots'
    settings.results_path = '/data/2024/DMTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/2024/DMTrack/output'
    settings.segmentation_path = '/root/DMTrack/output/test/segmentation_results'
    settings.tc128_path = '/root/DMTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/DMTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/DMTrack/data/trackingnet'
    settings.uav_path = '/root/DMTrack/data/uav'
    settings.vot18_path = '/root/DMTrack/data/vot2018'
    settings.vot22_path = '/root/DMTrack/data/vot2022'
    settings.vot_path = '/root/DMTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

