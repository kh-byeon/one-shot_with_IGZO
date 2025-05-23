import os
#from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, Subtract, Lambda, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2


import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from omniglot_loader import OmniglotLoader
#from modified_sgd import Modified_SGD


class EncoderNetwork:
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
        self.model = []
        self.learning_rate = learning_rate
        self.margin= margin
        self.batch_size=batch_size
        self.omniglot_loader = OmniglotLoader(
            dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
        self._construct_siamese_architecture(learning_rate_multipliers,
                                              l2_regularization_penalization)
        self.contrastive_loss_w_margin(margin)

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
        """
        
        # Let's define the cnn architecture        
        #Encoder
        encoder_net = Sequential()
        encoder_net.add(Dense(units=5000, activation='sigmoid',input_shape=self.input_shape,kernel_regularizer=l2(l2_regularization_penalization['Dense1']),name='Dense1'))
        encoder_net.add(Dense(units=2500, activation='sigmoid',kernel_regularizer=l2(l2_regularization_penalization['Dense2']),name='Dense2'))
        encoder_net.add(Dense(units=500, activation='sigmoid',kernel_regularizer=l2(l2_regularization_penalization['Dense3']),name='Dense3'))
        encoder_net.add(Dense(units=100, activation='sigmoid',kernel_regularizer=l2(l2_regularization_penalization['Dense4']),name='Dense4'))


        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)          

        encoded_image_1 = encoder_net(input_image_1)
        encoded_image_2 = encoder_net(input_image_2)
        
        # distance
        ###############################################################################
        #l2_distance_layer = Lambda(lambda tensors : tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]),keepdims=True)))
        l2_distance_layer = Lambda(lambda tensors : tf.expand_dims(K.sqrt(K.sum(K.square(tensors[0]-tensors[1]),axis=-1)),axis=-1))
        l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])
        #print('l2d',l2_distance)
        # Same class or not prediction
        #prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        #prediction = l2_distance
        #print('pred',prediction)
        self.model = Model(inputs=[input_image_1, input_image_2], outputs=l2_distance)


        # Define the optimizer and compile the model
        #optimizer = Modified_SGD(
            #lr=self.learning_rate,
            #lr_multipliers=learning_rate_multipliers,
            #momentum=0.5)

        self.model.compile(loss=self.contrastive_loss_w_margin(self.margin),metrics=['kullback_leibler_divergence']  , optimizer='Adam')

    


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
        train_accuracies = np.zeros(shape=(validate_each))
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
            train_loss, train_accuracy = self.model.train_on_batch(images, labels)
            #pred=self.model.predict_on_batch(images)
            #pred=self.model.predict_on_batch(images)
            #print('pred:',pred.shape)
            """
            tmp=np.array(images)
            print('\n train labels',labels.shape)
            print('img size0 :',tmp.shape)
            """
            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                self.model.optimizer.lr = self.model.optimizer.lr * 0.99
                #K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.99)
            #if K.get_value(self.model.optimizer.momentum) < final_momentum:
            #if self.model.optimizer.momentum < final_momentum:
                #K.set_value(self.model.optimizer.momentum, K.get_value(
                    #self.model.optimizer.momentum) + momentum_slope)
                #self.model.optimizer.momentum = self.model.optimizer.momentum + momentum_slope

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy
            
            # validation set
            # validate per 500 iter
            # evaluate per 1000 iter
            
            count += 1
            
            """
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(
                      self.model.optimizer.lr)))
            """
            print('Iteration %d/%d: Train loss: %f,lr = %f' %(iteration + 1, number_of_iterations, train_loss, K.get_value(self.model.optimizer.lr)))
            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            if (iteration + 1) % validate_each == 0:
                number_of_runs_per_alphabet = 40
                # use a support set size equal to the number of character in the alphabet
                validation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model, support_set_size, number_of_runs_per_alphabet, is_validation=True)
                valid_accuracies[val_count]=validation_accuracy
                val_count +=1
                self.train_loss_write_logs_to_tensorboard(iteration, train_losses, validate_each)
                count = 0

                # Some hyperparameters lead to 100%, although the output is almost the same in 
                # all images. 
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration
                        
                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        self.model.save_weights('models/' + model_name + '.h5')
            if (iteration +1) %1000 == 0:
                cur_iter=str(iteration+1)
                model_json = self.model.to_json()
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                with open('models/' + model_name +'_iter_'+cur_iter+ '.json', "w") as json_file:
                    json_file.write(model_json)
                self.model.save_weights('models/' + model_name + '.h5')

            if (iteration +1) % evaluate_each == 0:
                evaluation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model, support_set_size, 40, is_validation=False)
                eval_accuracies[eval_count]=evaluation_accuracy
                eval_count += 1

            # If accuracy does not improve for 10000 batches stop the training
            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' +
                      str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break
        
        self.valid_acc_write_logs_to_tensorboard( iteration,  valid_accuracies, val_count,eval_accuracies,eval_count)

        print('Trained Ended!')
        return best_validation_accuracy
