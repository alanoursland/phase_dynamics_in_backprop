from exp_util import run_experiment

if __name__ == '__main__':
    # Run the experiment using the exp_util API
    run_experiment(dataset_name='MNIST', optimizer_name='Adam', experiment_name='mnist_adam_mlp2', num_runs=20)
