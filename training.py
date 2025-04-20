# -*- coding: utf-8 -*-
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
import shutil
import csv
import tensorflow as tf
import numpy as np
import multiprocessing
from pd_siam_net import SiameseNetwork

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

multiprocessing.set_start_method("spawn", force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_experiment(lr, margin, file_lock):
    print(f"Starting experiment with lr={lr}, margin={margin} on GPU 0")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    dataset_path = '../omniglot/python'
    use_augmentation = True
    batch_size = 32

    # Learning Rate multipliers
    learning_rate_multipliers = {f'Conv{i+1}': 1 for i in range(6)}
    learning_rate_multipliers.update({'Dense1': 1, 'Dense2': 1})

    # L2 Regularization
    l2_penalization = {f'Conv{i+1}': 0 for i in range(6)}
    l2_penalization.update({'Dense1': 0, 'Dense2': 0})

    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=lr,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        margin=margin
    )

    support_set_size = 5
    validate_each = 1000
    evaluate_each = 1000
    number_of_train_iterations = 1000000

    directories = ['./models', './logs', './bestmodel', './lrsweep']
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 

    validation_accuracy, acc_list = siamese_network.train_siamese_network(
        number_of_iterations=number_of_train_iterations,
        support_set_size=support_set_size,
        validate_each=validate_each, 
        evaluate_each=evaluate_each,
        model_name=f'siamese_net_lr{lr}_margin{margin}'
    )

    result_file = './lrsweep/result.txt'
    
    with file_lock:
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write("=== Learning Rate Sweep Results ===\n")

        with open(result_file, 'a') as f:
            f.write(f'|| lr={lr} margin={margin} final acc={validation_accuracy} best_valid_acc={validation_accuracy} ||\n')

    print(f'Final Evaluation Accuracy for lr={lr}, margin={margin} = {validation_accuracy}')


if __name__ == "__main__":
    lr_list = [5e-3]
    margin_list = [ 7]

    with multiprocessing.Manager() as manager:  
        file_lock = manager.Lock() 

        processes = []
        for i, (lr, margin) in enumerate(zip(lr_list, margin_list)):
            p = multiprocessing.Process(target=run_experiment, args=(lr, margin, file_lock))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All experiments completed!")
