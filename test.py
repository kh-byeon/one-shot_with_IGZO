from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import os,shutil
#from modified_siamese_network import SiameseNetworkforTCAM
#from encoder_network import EncoderNetwork
from vgg_19 import SiameseNetworkforTCAM
from random_projection import RandomProj
import tensorflow as tf
import tensorflow.keras.backend as K
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

def main(lr):
    dataset_path = '../omniglot/python'
    use_augmentation = True
    learning_rate = lr
    batch_size = 1
    margin= 2
      
    
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
    
    #siamese_network = VGG_19(
    siamese_network = SiameseNetworkforTCAM(    
    #siamese_network = EncoderNetwork(
    #siamese_network = RandomProj(
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
    siamese_network.model_test.load_weights('./models/siamese_net.h5')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    #shutil.rmtree('./models/')
    #shutil.rmtree('./logs/')
    siamese_network.omniglot_loader.split_train_datasets()
    
    test_images=siamese_network.omniglot_loader.get_test_image()
    predictions=siamese_network.model_test.predict_on_batch(test_images)
    
    #print('pred',predictions)
    tmp=np.array(predictions)
    print('pred size',tmp.shape)

    pred0=predictions[0]
    pred1=predictions[1]
    pred2=predictions[2]
    tmp1=np.array(pred0)
    print('pred0 size',tmp1.shape)
    same_char_dist=K.sum(K.square(pred0-pred1))
    dif_char_dist=K.sum(K.square(pred0-pred2))
    p=1
    print('dist btw same char',same_char_dist/p)
    print('dist btw dif char',dif_char_dist/p)


    """
    flat0=tf.expand_dims(tf.reshape(tmp[0],[105*105]),axis=0)
    flat1=tf.expand_dims(tf.reshape(tmp[1],[105*105]),axis=0)
    flat2=tf.expand_dims(tf.reshape(tmp[2],[105*105]),axis=0)
    print('original vector',flat0.shape)
            
    ### before random projection ###
    #same_char_dist=K.sqrt(K.sum(K.square(flat0-flat1)))
    #dif_char_dist=K.sqrt(K.sum(K.square(flat0-flat2)))
    same_char_dist=K.sum(K.square(flat0-flat1))
    dif_char_dist=K.sum(K.square(flat0-flat2))
    """
    
if __name__ == "__main__":
    #lr_list=[100,50,10,1,1e-2,1e-4,1e-6,1e-9]
    #lr_list=[1,1e-2,1e-4,1e-6,1e-9]
    if not os.path.exists('./lrsweep'):
        os.makedirs('./lrsweep')
    shutil.rmtree('./lrsweep/')

    main(1)
    #for sweep in lr_list:
        #main(sweep)
    print('type tensorboard --logdir=./logs/   for tensorboard')
