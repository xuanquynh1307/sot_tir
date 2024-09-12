from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.lsotb_path = 'D:/Case-Study/CSWinTT/data/LSOTB_TIR'
    settings.network_path = r'F:\single-object-tracking-tir\output\test/networks'
    settings.prj_dir = r'F:\single-object-tracking-tir'
    settings.result_plot_path = r'F:\single-object-tracking-tir\output\test/result_plots'
    settings.results_path = r'F:\single-object-tracking-tir\output\test/tracking_results'
    settings.save_dir = r'F:\single-object-tracking-tir\output'
    settings.segmentation_path = r'F:\single-object-tracking-tir\output\test/segmentation_results'

    return settings
