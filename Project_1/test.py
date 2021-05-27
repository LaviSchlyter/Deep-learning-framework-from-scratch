import experiment_report
from run_experiments import run_experiments


def main():
    EXPERIMENTS = [experiment_report.EXPERIMENT_CONV_SHARED_AUX_MORE]
    run_experiments("test", 3, plot_titles=True, plot_loss=True, experiments=EXPERIMENTS)


if __name__ == '__main__':
    main()
