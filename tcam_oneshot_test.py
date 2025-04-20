from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import os,shutil
import csv
from IGZO_TCAM_training import SiameseNetworkforTCAM
#from pre_trained_modified_siamese_network import SiameseNetworkforTCAM
#from vgg_19 import SiameseNetworkforTCAM
#from modified_siamese_network import SiameseNetworkforTCAM
import tensorflow as tf
#from encoder_network import EncdoerNetwork
import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:    
    print(e)

def main():
    dataset_path = '../omniglot/python'
    use_augmentation = True
    learning_rate = 1
    batch_size = 32
    margin= 1
      
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Conv5'] = 1
    learning_rate_multipliers['Conv6'] = 1
    learning_rate_multipliers['Dense1'] = 1
    learning_rate_multipliers['Dense2'] = 1

    #l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 0
    l2_penalization['Conv2'] = 0
    l2_penalization['Conv3'] = 0
    l2_penalization['Conv4'] = 0
    l2_penalization['Conv5'] = 0
    l2_penalization['Conv6'] = 0
    l2_penalization['Dense1'] =0
    l2_penalization['Dense2'] = 0
    

    
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net'
    summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
    #siamese_network = VGG_19(
    siamese_network = SiameseNetworkforTCAM(    
    #siamese_network = EncoderNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path,
        margin=margin
    )

    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 5
    validate_each = 1000
    evaluate_each = 2000
    number_of_train_iterations = 1#000000
    #siamese_network.model_test.load_weights('./model_from_308/92.h5')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    #shutil.rmtree('./models/')
    #shutil.rmtree('./logs/')

    
    siamese_network.omniglot_loader.split_train_datasets()
    
    #siamese_network.omniglot_loader.TCAM_test(siamese_network.model_test,
                                                                        #support_set_size, 1, False)

    number_of_iterations=1
    best_eval_acc=0
    eval_count=0
    eval_accuracies = np.zeros(shape=(int (number_of_iterations)))

    #siamese_network.omniglot_loader.one_shot_test_Sim_Mem(siamese_network.model_test, support_set_size, 1, False)
    siamese_network.omniglot_loader.one_shot_test(siamese_network.model_test, support_set_size, 1, False)

    """
    for iteration in range(number_of_iterations):
        evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test_Sim_Mem(siamese_network.model_test,
                                                                        support_set_size, 40, False)
                                                                        #support_set_size, 4, False)
      
        eval_accuracies[eval_count]=evaluation_accuracy
        eval_count +=1

        if evaluation_accuracy > best_eval_acc:
            best_eval_acc = evaluation_accuracy
        print('Best Evaluation Accuracy = ' + str(best_eval_acc))    

        acc_write_logs_to_tensorboard(eval_accuracies,number_of_iterations)
    """

    """
    if not os.path.exists('./lrsweep'):
        os.makedirs('./lrsweep')
    with open('./lrsweep/result.txt','a') as F:
        F.write('|| ')
        F.write('lr='+str(lr)+' '+'m='+str(margin)+' final acc='+str(evaluation_accuracy)+'best_eval_acc= '+str(validation_accuracy)+' ||')
        F.write('\n')
    """
  
def acc_write_logs_to_tensorboard(eval_accuracies,eval_count):  
    for index in range(0, eval_count):
        with summary_writer.as_default():
            tf.summary.scalar('One-shot evaluation Accuracy',eval_accuracies[index],step=index)      


if __name__ == "__main__":
    main()
    """
    for sweep in lr_list:
        main(sweep)
    """
    print('type tensorboard --logdir=./logs/   for tensorboard')


