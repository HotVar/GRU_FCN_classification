# Bayesian Optimization tools
from bayes_opt import BayesianOptimization
from train import train

def optimize_hparams():
    # Bounded region of parameter space
    pbounds = {'BATCH_SIZE_INDEX': (0, 2),
               'N_EPOCHS': (20, 300),
               'AUG_RATE': (0, 1),
               'hidden_size':(2, 16),
               'num_layers':(1, 8),
               'dropout_rate':(0, 1),
               'learning_rate':(1e-06, 1e-02)}

    optimizer = BayesianOptimization(f=train,
                                     pbounds=pbounds,
                                     random_state=42)
    optimizer.maximize(init_points=3,
                       n_iter=10)

    print(optimizer.max)
    with open('optimized_params.txt', 'a') as f:
        for k, v in optimizer.max.items():
            f.write(f'{k} : {v}\n')


if __name__ == '__main__':
    optimize_hparams()