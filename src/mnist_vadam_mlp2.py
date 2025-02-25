from exp_util import run_experiment

if __name__ == '__main__':
    # Run the experiment using the exp_util API
    run_experiment(dataset_name='MNIST', optimizer_name='Vadam', experiment_name='mnist_vadam_mlp2', num_runs=20)
