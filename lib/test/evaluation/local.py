from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.lsotb_path = 'C:/Users/PC/Documents/DC/01-Datasets/LSOTB-TIR-2023'
    settings.network_path = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir/networks'
    settings.prj_dir = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir'
    settings.result_plot_path = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir/output/test/result_plots'
    settings.results_path = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir/output/test/tracking_results'
    settings.save_dir = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir/output'
    settings.segmentation_path = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir/output/test/segmentation_results'

    return settings
