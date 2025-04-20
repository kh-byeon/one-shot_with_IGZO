
import os
#from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from pathlib import Path

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
np.set_printoptions(precision=6, suppress=False)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
from t_omniglot_loader import OmniglotLoader
#from modified_sgd import Modified_SGD
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Netlist import Circuit



class SiameseNetworkforCurrent_training:
    """Class that constructs the Siamese Net for training

    This Class was constructed to create the siamese net and train it.

    Attributes:
        input_shape: image size
        model: current siamese model
        learning_rate: SGD learning rate
        omniglot_loader: instance of OmniglotLoader
        summary_writer: tensorflow writer to store the logs
    """

    def __init__(self, dataset_path,  learning_rate, batch_size, use_augmentation,
                 learning_rate_multipliers, l2_regularization_penalization, tensorboard_log_path, margin):
        """Inits SiameseNetwork with the provided values for the attributes.

        It also constructs the siamese network architecture, creates a dataset 
        loader and opens the log file.

        Arguments:
            dataset_path: path of Omniglot dataset    
            learning_rate: SGD learning rate
            batch_size: size of the batch to be used in training
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not
            learning_rate_multipliers: learning-rate multipliers (relative to the learning_rate
                chosen) that will be applied to each fo the conv and dense layers
                for example:
                    # Setting the Learning rate multipliers
                    LR_mult_dict = {}
                    LR_mult_dict['conv1']=1
                    LR_mult_dict['conv2']=1
                    LR_mult_dict['dense1']=2
                    LR_mult_dict['dense2']=2
            l2_regularization_penalization: l2 penalization for each layer.
                for example:
                    # Setting the Learning rate multipliers
                    L2_dictionary = {}
                    L2_dictionary['conv1']=0.1
                    L2_dictionary['conv2']=0.001
                    L2_dictionary['dense1']=0.001
                    L2_dictionary['dense2']=0.01
            tensorboard_log_path: path to store the logs                
        """
        self.input_shape = (105, 105, 1)  # Size of images
        self.model_train = []
        self.model_test = []
        self.learning_rate = learning_rate
        self.margin= margin
        self.batch_size=batch_size
        self.omniglot_loader = OmniglotLoader(
            dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
        self._construct_siamese_architecture(learning_rate_multipliers,
                                              l2_regularization_penalization)
        self.contrastive_loss_w_margin(margin)
        #self.distance_circuit(img1,img2)

    def contrastive_loss_w_margin(self,margin):
        def contrastive_loss(y_true, y_pred):
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred,0))
            return (1/2)*(y_true * square_pred + (1-y_true)*margin_square)
        return contrastive_loss
    

    def _construct_siamese_architecture(self, learning_rate_multipliers,
                                         l2_regularization_penalization):
        """ Constructs the siamese architecture and stores it in the class

        Arguments:
            learning_rate_multipliers
            l2_regularization_penalization

        modified for IGZO TCAM
        """
        
        # Let's define the cnn architecture
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=64, kernel_size=(13, 13),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv1']),
                                     name='Conv1'))       
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Conv2D(filters=64, kernel_size=(11, 11),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv2']),
                                     name='Conv2'))
        
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Conv2D(filters=64, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv3']),
                                     name='Conv3'))        
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Conv2D(filters=128, kernel_size=(5, 5),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv4']),
                                     name='Conv4'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Flatten())

        convolutional_net.add(Dense(units=1000, activation='sigmoid',kernel_regularizer=l2(l2_regularization_penalization['Dense1']),name='Dense1'))       
        convolutional_net.add(Dense(units=100, activation='sigmoid',name='Dense3'))
        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)              

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)
        
        # distance
        ###############################################################################
        #l2_distance_layer = Lambda(lambda tensors : tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]),keepdims=True)))

        #l2_distance_layer = Lambda(lambda tensors : tf.expand_dims(K.sqrt(K.sum(K.square(tensors[0]-tensors[1]),axis=-1)),axis=-1))
        #l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])
        #sim_mem_dist=self.distance_circuit(encoded_image_1,encoded_image_2)
        #emb=tf.concat([encoded_image_1,encoded_image_2],axis=0)
        #emb_tmp=tf.concat([encoded_image_1,encoded_image_2],axis=0)
        #print('emb',emb_tmp.shape)
        coef=1.0
        """
        sim_mem_distance_layer = Lambda(lambda tensors: tf.expand_dims(
            K.sum(
                (-1.35171722763369e-05) * tf.pow(coef * (tensors[0] - tensors[1]), 5) +
                (2.91306114091758e-05) * tf.pow(coef * (tensors[0] - tensors[1]), 4) +
                (2.09437682432142e-05) * tf.pow(coef * (tensors[0] - tensors[1]), 3) +
                (1.02187733012733e-05) * tf.pow(coef * (tensors[0] - tensors[1]), 2) +
                (-6.18938530611747e-06) * coef * (tensors[0] - tensors[1]) +
                (2.62644481443442e-07), axis=-1
            ), axis=-1))
        """

        sim_mem_distance_layer = Lambda(lambda tensors: tf.expand_dims(
            K.sum(
                (6.07639247315868E-13 ) * tf.pow(coef * (tensors[0] - tensors[1]), 5) +
                (3.39122644532955E-09) * tf.pow(coef * (tensors[0] - tensors[1]), 4) +
                (-5.65000484793245E-12) * tf.pow(coef * (tensors[0] - tensors[1]), 3) +
                (2.23468460860643E-07) * tf.pow(coef * (tensors[0] - tensors[1]), 2) +
                (-1.48889581689055E-11) * coef * (tensors[0] - tensors[1]) +
                (-8.75590286668583E-08), axis=-1
            ), axis=-1))
                

        sim_mem_distance=sim_mem_distance_layer([encoded_image_1*3, encoded_image_2*3])
        #sim_mem_distance=tf.expand_dims(sim_mem_distance,axis=0)
        #print("sim_mem_distance",sim_mem_distance)
        #sim_mem_dist=tf.py_function(func=self.distance_circuit, inp=[encoded_image_1,encoded_image_2],Tout=tf.float32)
        #print(sim_mem_dist)
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=sim_mem_distance*100000)

        prediction =encoded_image_1
        self.model_test = Model(inputs=input_image_1, outputs=prediction)


        # Define the optimizer and compile the model
        #optimizer = Modified_SGD(
            #lr=self.learning_rate,
            #lr_multipliers=learning_rate_multipliers,
            #momentum=0.5)

        self.model_train.compile(loss=self.contrastive_loss_w_margin(self.margin) , optimizer='SGD')

    

    def train_loss_write_logs_to_tensorboard(self, current_iteration, train_losses, validate_each):
        """ Writes the logs to a tensorflow log file

        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so 
        instead we write the logs every evaluate_each iteration.

        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the current one-shot task in the 
                validation set
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
        """
        #summary = tf.Summary()
        # Write to log file the values from the last evaluate_every iterations
        for index in range(0, validate_each):
            with self.summary_writer.as_default():
                tf.summary.scalar('Train Loss',train_losses[index], step=index)         
            

    def valid_acc_write_logs_to_tensorboard( self, iteration,  valid_accuracies, val_count,eval_accuracies,eval_count):        
             # Write to log file the values from the last evaluate_every iterations
        for index in range(0, val_count):            
            with self.summary_writer.as_default():
                tf.summary.scalar('One-shot validation Accuracy',valid_accuracies[index],step=index)            
        for index in range(0, eval_count):            
            with self.summary_writer.as_default():
                tf.summary.scalar('One-shot evaluation Accuracy',eval_accuracies[index],step=index)          
                   

    def train_siamese_network(self, number_of_iterations, support_set_size,
                              final_momentum, momentum_slope, validate_each,
                              evaluate_each, model_name):
        """ Train the Siamese net

        This is the main function for training the siamese net. 
        In each every validate_each train iterations we evaluate one-shot tasks in 
        validation and evaluation set. We also write to the log file.

        Arguments:
            number_of_iterations: maximum number of iterations to train.
            support_set_size: number of characters to use in the support set
                in one-shot tasks.
            final_momentum: mu_j in the paper. Each layer starts at 0.5 momentum
                but evolves linearly to mu_j
            momentum_slope: slope of the momentum evolution. In the paper we are
                only told that this momentum evolves linearly. Because of that I 
                defined a slope to be passed to the training.
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
            model_name: save_name of the model

        Returns: 
            Evaluation Accuracy
        """

        # First of all let's divide randomly the 30 train alphabets in train
        # and validation with 24 for training and 6 for validation
        self.omniglot_loader.split_train_datasets()

        # Variables that will store 100 iterations losses and accuracies
        # after validate_each iterations these will be passed to tensorboard logs
        train_losses = np.zeros(shape=(validate_each))
        #train_accuracies = np.zeros(shape=(validate_each))
        count = 0
        earrly_stop = 0
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0
        val_count=0
        eval_count=0
        valid_accuracies = np.zeros(shape=(int(number_of_iterations/validate_each)))
        eval_accuracies = np.zeros(shape=(int(number_of_iterations/evaluate_each)))
        # Train loop
        for iteration in range(number_of_iterations):

            # train set
            images, labels = self.omniglot_loader.get_train_batch() # same class - label =1
            
            # Example using tf.data
            #train_dataset = tf.data.Dataset.from_generator(lambda: self.omniglot_loader, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None, 105, 105, 1]), tf.TensorShape([None,])))
            #train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            #images, labels =train_dataset
            #images=np.array(images)
            #labels=np.array(labels)
            #print(f'Images shape: {images.shape}')
            #print(f'Labels shape: {labels.shape}')

            #train_loss, train_accuracy = self.model_train.train_on_batch(images, labels)
            #train_loss = self.model_train.train_on_batch(train_dataset)
            
            train_loss = self.model_train.train_on_batch(images, labels)
            #print('train loss',train_loss)
            #tmp=np.array(images)
            #print(tmp.shape)
            #pred=self.model_train.predict_on_batch(images)  #2,64,~
            #print('pred',pred)
            #print('pred size',pred.shape)
            #print('pred_dist: %.5e'%(pred[0]))
            #print('pred_dist: %.5e'%(pred[1]))
            #print('Iteration %d/%d: Train loss: %f,lr = '%(iteration + 1, number_of_iterations, train_loss)+format(K.get_value(self.learning_rate),".3E"))


            """
            tmp=np.array(images)
            print('\n train labels',labels.shape)
            print('img size0 :',tmp.shape)
            """
            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                self.learning_rate = self.learning_rate * 0.99
                #K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.99)
            #if K.get_value(self.model.optimizer.momentum) < final_momentum:
            #if self.model.optimizer.momentum < final_momentum:
                #K.set_value(self.model.optimizer.momentum, K.get_value(
                    #self.model.optimizer.momentum) + momentum_slope)
                #self.model.optimizer.momentum = self.model.optimizer.momentum + momentum_slope

            train_losses[count] = train_loss
            #train_accuracies[count] = train_accuracy
            
            # validation set
            # validate per 500 iter
            # evaluate per 1000 iter
            
            count += 1
                       
            print('Iteration %d/%d: Train loss: %f,lr = '%(iteration + 1, number_of_iterations, train_loss)+format(K.get_value(self.learning_rate),".3E"))
            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            if (iteration + 1) % validate_each == 0:
                number_of_runs_per_alphabet = 40
                # use a support set size equal to the number of character in the alphabet
                validation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model_test, support_set_size, number_of_runs_per_alphabet, is_validation=True)
                valid_accuracies[val_count]=validation_accuracy
                val_count +=1
                self.train_loss_write_logs_to_tensorboard(iteration, train_losses, validate_each)
                count = 0

                # Some hyperparameters lead to 100%, although the output is almost the same in 
                # all images. 
                if (validation_accuracy == 1.0 ):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_validation_accuracy))
                    break
                    #return 0                
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration
                        
                        model_json = self.model_test.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        self.model_test.save_weights('models/' + model_name + '.h5')
            if (iteration +1) %1000 == 0:
                cur_iter=str(iteration+1)
                model_json = self.model_test.to_json()
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                with open('models/' + model_name +'_iter_'+cur_iter+ '.json', "w") as json_file:
                    json_file.write(model_json)
                self.model_test.save_weights('models/' + model_name + '.h5')

            if (iteration +1) % evaluate_each == 0:
                evaluation_accuracy = self.omniglot_loader.one_shot_test_fitted_simmem(
                    self.model_test, support_set_size, 40, is_validation=False)
                eval_accuracies[eval_count]=evaluation_accuracy
                eval_count += 1

            # If accuracy does not improve for 10000 batches stop the training
            if iteration - best_accuracy_iteration > 20000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' +
                      str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break
        
        self.valid_acc_write_logs_to_tensorboard( iteration,  valid_accuracies, val_count,eval_accuracies,eval_count)

        print('Trained Ended!')
        return best_validation_accuracy,eval_accuracies
