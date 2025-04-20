import os
#from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
import gc

import tensorflow as tf
import csv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from t_omniglot_loader import OmniglotLoader
#from modified_sgd import Modified_SGD


class SiameseNetworkforTCAM:
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
        convolutional_net.add(Dense(units=2048, activation='sigmoid',name='Dense3'))
        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)              

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)
        
        # distance
        ###############################################################################
        #l2_distance_layer = Lambda(lambda tensors : tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]),keepdims=True)))
        l2_distance_layer = Lambda(lambda tensors : tf.expand_dims(K.sqrt(K.sum(K.square(tensors[0]-tensors[1]),axis=-1)),axis=-1))
        l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])
        #print('l2d',l2_distance)
        # Same class or not prediction
        #prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        prediction =encoded_image_1
        #print('pred',prediction)
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=l2_distance)
        self.model_test = Model(inputs=input_image_1, outputs=prediction)


        # Define the optimizer and compile the model
        #optimizer = Modified_SGD(
            #lr=self.learning_rate,
            #lr_multipliers=learning_rate_multipliers,
            #momentum=0.5)

        self.model_train.compile(loss=self.contrastive_loss_w_margin(self.margin),metrics=['kullback_leibler_divergence']  , optimizer='SGD')

    


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
            

    def valid_acc_write_logs_to_tensorboard( self,eval_accuracies,eval_count):        
             # Write to log file the values from the last evaluate_every iterations
                 
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
        best_eval_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0
        evaluation_accuracy=0.0
        val_count=0
        eval_count=0
        valid_accuracies = np.zeros(shape=(int(number_of_iterations/validate_each)))
        eval_accuracies = np.zeros(shape=(int(number_of_iterations/evaluate_each)))
        # Train loop
        for iteration in range(number_of_iterations):
        #for iteration in range(33001,number_of_iterations-1,1):

            # train set
            #images, labels = self.omniglot_loader.mod_get_train_batch() # same class - label =1
            images, labels = self.omniglot_loader.get_train_batch() # same class - label =1
            train_loss, train_accuracy = self.model_train.train_on_batch(images, labels)
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
                self.learning_rate = self.learning_rate * 0.99
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
                       
            print('Iteration %d/%d: Train loss: %f,lr = '%(iteration + 1, number_of_iterations, train_loss)+format(K.get_value(self.learning_rate),".3E"))
            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            if (iteration +1) % evaluate_each == 0:
                evaluation_accuracy = self.omniglot_loader.one_shot_test_TCAM(
                #evaluation_accuracy = self.omniglot_loader.one_shot_test_Sim_Mem(
                #evaluation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model_test, support_set_size, 40, is_validation=False)
                eval_accuracies[eval_count]=evaluation_accuracy
                eval_count += 1
                c=open('embsize_2048.csv','w')
                wr=csv.writer(c)
                wr.writerow(eval_accuracies)

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
                if (validation_accuracy == 1.0):# and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_eval_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if evaluation_accuracy > best_eval_accuracy:
                        best_eval_accuracy = evaluation_accuracy
                        best_accuracy_iteration = iteration
                        
                        model_json = self.model_test.to_json()
                        cur_iter=str(iteration+1)

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
                self.model_test.save_weights('models/' + model_name+'_iter_'+cur_iter + '.h5')
            
            

                #./models/siamese_net_iter_259000.h5

            # If accuracy does not improve for 10000 batches stop the training
            #if iteration - best_accuracy_iteration-129000 > 20000:
            if iteration - best_accuracy_iteration > 20000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' +
                      str(best_eval_accuracy))
                print('Validation Accuracy = ' + str(best_eval_accuracy))
                break
        
        self.valid_acc_write_logs_to_tensorboard(eval_accuracies,eval_count)

        #self.model_test.summary()
        #self.model_test.layers
        """
        print('size',len(self.model_test.get_weights()[0][0]))
        print('size',len(self.model_test.get_weights()[0][1]))
        print('size',len(self.model_test.get_weights()[1]))
        print('size',len(self.model_test.get_weights()[2]))
        print('size',len(self.model_test.get_weights()[3]))
        """
        #w_dense=self.model_test.layers.get_weights()[8]

        print('Trained Ended!')
        
        '''
        vis_data=self.omniglot_loader.visualization()
        emb=self.model_test.predict_on_batch(vis_data)
        #print('emb',len(emb[0]))       
        gaus=tf.random.normal((100,3))
        data=tf.matmul(emb,gaus)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30,50)

        marker_list=['o','o','o','o','o','o','o','o','s','s','s','s','s','s','s','s','h','h','h','h','h','h','h','h','D']       
        color_list=['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w','b']   
        print('len',int(len(data)/5))
        for i in range(0, int(len(data)/5)):           
            for same in range(0,5):
                ax.scatter(data[i*5+same][0],data[i*5+same][1],data[i*5+same][2],marker=marker_list[i], c=color_list[i])
                
        #print('mak',plt.markers)
        plt.savefig('test.png')
        print('embedding printed')
        '''
        return best_eval_accuracy, eval_accuracies
