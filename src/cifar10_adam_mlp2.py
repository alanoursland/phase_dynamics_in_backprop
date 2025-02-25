from exp_util import run_experiment

if __name__ == '__main__':
    run_experiment(
        dataset_name='CIFAR10',
        optimizer_name='Adam',
        experiment_name='cifar10_aadam_cnn',
        num_runs=20
    )
