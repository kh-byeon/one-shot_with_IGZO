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
np.set_printoptions(precision=3, suppress=True)
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
        self.distance_circuit(img1,img2)

    def contrastive_loss_w_margin(self,margin):
        def contrastive_loss(y_true, y_pred):
            #square_pred = K.square(y_pred)
            #sim_mem_dist = self.distance_circuit(enc_img1,enc_img2)
            #margin_square = K.square(K.maximum(margin - y_pred,0))
            margin_square = K.square(K.maximum(margin - y_pred,0))
            return (1/2)*(y_true * K.square(y_pred) + (1-y_true)*margin_square)
        return contrastive_loss

    def distance_circuit(self,img1,img2):
        circuit=Circuit('dist')
        directory_path = Path(__file__).resolve().parent
        netlist_path = directory_path.joinpath('sim_mem_dist_cell.cir')
        parser = SpiceParser(path=str(netlist_path))
        circuit = parser.build_circuit()
        
        sl_array= (np.array(img1)-0.5)*2
        support_data= -(np.array(img2)*2)+1.6
        for	waynum in range(0,1):
            for cellnum in range(0,100):            
                sl_vth='sl_way'+str(waynum)+'_cell'+str(cellnum)
                gate_node='gate_way'+str(waynum)+'_cell'+str(cellnum)
                inter='inter_way'+str(waynum)+'_cell'+str(cellnum)
                vgate='vgate_way'+str(waynum)+'_cell'+str(cellnum)

                circuit.V(sl_vth,gate_node,inter, str(sl_array[cellnum]))
                circuit.V(vgate,inter,'GND', str(vcap[cellnum]))
            circuit.V(ml,ml, 'GND', '1.2')
        simulator = circuit.simulator()
        #simulator = circuit.simulator(temperature=25, nominal_temperature=25,simulator='ngspice-subprocess')
        analysis = simulator.operating_point()
        dist_current=float(-analysis.Vml_0)
        print('training dist current:',dist_current)
        return dist_current

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
        sim_mem_dist=self.distance_circuit(encoded_image_1,encoded_image_2)


        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=sim_mem_dist)

        prediction =encoded_image_1
        self.model_test = Model(inputs=input_image_1, outputs=prediction)


        # Define the optimizer and compile the model
        #optimizer = Modified_SGD(
            #lr=self.learning_rate,
            #lr_multipliers=learning_rate_multipliers,
            #momentum=0.5)

        self.model_train.compile(loss=self.contrastive_loss_w_margin(self.margin),metrics=['kullback_leibler_divergence']  , optimizer='SGD')

    
    '''
    def fitted_function(self, x):
        ans=(6.764e-8)*x**5+(1.752e-06)*x**4+(-8.159e-7)*x**3+(-2.362e-7)*x**2+(7.474e-8)*x+(4.201e-9) 
        #print('ans',ans)
        return ans
    '''

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

            #pred0,pred1=self.model_train.predict_on_batch(images)
            #print('pred0:',pred0)
            #print('pred1:',pred1)
            #print('sub',100*(pred0-pred1))

            train_loss, train_accuracy = self.model_train.train_on_batch(images, labels)
            
            #pred=self.model_test.predict_on_batch(images)
            #print('pred',pred)
            


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
                evaluation_accuracy = self.omniglot_loader.one_shot_test_Sim_Mem(
                    self.model_test, support_set_size, 40, is_validation=False)
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
        return best_validation_accuracy,eval_accuracies





=================================================================
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
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
np.set_printoptions(precision=3, suppress=True)
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

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
 

def distance_circuit(img1,img2):
    img1=img1.numpy()
    img2=img2.numpy()
    print('img1',img1.shape)

    circuit=Circuit('dist')
    directory_path = Path(__file__).resolve().parent
    netlist_path = directory_path.joinpath('sim_mem_dist_cell.cir')
    parser = SpiceParser(path=str(netlist_path))
    circuit = parser.build_circuit()
    circuit.include('../libraries/ptm_45nm.txt')

    sl_array= (np.array(img1)-0.5)*2
    support_data= -(np.array(img2)*2)+1.6
    for	waynum in range(0,32):
        ml='ML'+str(waynum)
        sl_tmp=sl_array[waynum]
        sup_tmp=support_data[waynum]
        for cellnum in range(0,100):            
            sl_vth='sl_way'+str(waynum)+'_cell'+str(cellnum)
            gate_node='gate_way'+str(waynum)+'_cell'+str(cellnum)
            inter='inter_way'+str(waynum)+'_cell'+str(cellnum)
            vgate='vgate_way'+str(waynum)+'_cell'+str(cellnum)

            circuit.V(sl_vth,gate_node,inter, str(sl_tmp[cellnum]))
            circuit.V(vgate,inter,'GND', str(sup_tmp[cellnum]))
        circuit.V(ml,ml, 'GND', '1.2')
    #simulator = circuit.simulator()
    simulator = circuit.simulator(temperature=25, simulator='ngspice-subprocess')
    analysis = simulator.operating_point()
    dist_current=[(float(-analysis.VML0)),(float(-analysis.VML1)),(float(-analysis.VML2)),(float(-analysis.VML3)),(float(-analysis.VML4)),(float(-analysis.VML5)),(float(-analysis.VML6)),(float(-analysis.VML7)),(float(-analysis.VML8)),(float(-analysis.VML9)),(float(-analysis.VML10)),(float(-analysis.VML11)),(float(-analysis.VML12)),(float(-analysis.VML13)),(float(-analysis.VML14)),(float(-analysis.VML15)),(float(-analysis.VML16)),(float(-analysis.VML17)),(float(-analysis.VML18)),(float(-analysis.VML19)),(float(-analysis.VML20)),(float(-analysis.VML21)),(float(-analysis.VML22)),(float(-analysis.VML23)),(float(-analysis.VML24)),(float(-analysis.VML25)),(float(-analysis.VML26)),(float(-analysis.VML27)),(float(-analysis.VML28)),(float(-analysis.VML29)),(float(-analysis.VML30)),(float(-analysis.VML31))]
    dist_tmp=np.array(dist_current)
    #print('dist_current',dist_tmp.shape)
    return dist_tmp

def contrastive_loss_w_margin(margin):
    def contrastive_loss(y_true, y_pred):
        #tf.print('yPred',tf.shape(y_pred))
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred,0))
        #margin_square = K.square(K.maximum(margin - sim_mem_dist,0))
        #return (1/2)*(y_true * K.square(sim_mem_dist) + (1-y_true)*margin_square)
        return (1/2)*(y_true * square_pred + (1-y_true)*margin_square)
    return contrastive_loss

convolutional_net = Sequential()
convolutional_net.add(Conv2D(filters=64, kernel_size=(13, 13),
                                activation='relu',
                                input_shape=(105, 105, 1),                                
                                name='Conv1'))       
convolutional_net.add(MaxPool2D())
convolutional_net.add(Conv2D(filters=64, kernel_size=(11, 11),
                                activation='relu',
                                name='Conv2'))
        
convolutional_net.add(MaxPool2D())
convolutional_net.add(Conv2D(filters=64, kernel_size=(7, 7),
                                activation='relu',
                                name='Conv3'))        
convolutional_net.add(MaxPool2D())
convolutional_net.add(Conv2D(filters=128, kernel_size=(5, 5),
                                activation='relu',
                                name='Conv4'))
convolutional_net.add(MaxPool2D())

convolutional_net.add(Flatten())

convolutional_net.add(Dense(units=1000, activation='sigmoid',name='Dense1'))       
convolutional_net.add(Dense(units=100, activation='sigmoid',name='Dense3'))
# Now the pairs of images
input_image_1 = Input((105, 105, 1))
input_image_2 = Input((105, 105, 1))              

encoded_image_1 = convolutional_net(input_image_1)
encoded_image_2 = convolutional_net(input_image_2)

sim_mem_dist=tf.py_function(func=distance_circuit, inp=[encoded_image_1,encoded_image_2],Tout=tf.float32)
print('sim mem size',sim_mem_dist)
model_train = Model(inputs=[input_image_1, input_image_2], outputs=sim_mem_dist)

        
model_train.compile(optimizer='sgd', loss=contrastive_loss_w_margin(margin=1))


omniglot_loader=OmniglotLoader(dataset_path='../omniglot/python', use_augmentation=True, batch_size=32)
omniglot_loader.split_train_datasets()
images, labels = omniglot_loader.get_train_batch()
#print('image',images.shape)
model_train.fit(images,labels, epochs=10)



========================================================================================

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
np.set_printoptions(precision=3, suppress=True)
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
            #tf.print('yPred',tf.shape(y_pred))
            square_pred = K.square(y_pred)
            #sim_mem_dist = tf.py_function(func=self.distance_circuit, inp=[y_pred[0],y_pred[1]],Tout=tf.float32)
            margin_square = K.square(K.maximum(margin - y_pred,0))
            #margin_square = K.square(K.maximum(margin - sim_mem_dist,0))
            #return (1/2)*(y_true * K.square(sim_mem_dist) + (1-y_true)*margin_square)
            return (1/2)*(y_true * square_pred + (1-y_true)*margin_square)
        return contrastive_loss

    def distance_circuit(self,img1,img2):
        img1=img1.numpy()
        img2=img2.numpy()
        print('img1',img1.shape)
        circuit=Circuit('dist')
        directory_path = Path(__file__).resolve().parent
        netlist_path = directory_path.joinpath('sim_mem_dist_cell.cir')
        parser = SpiceParser(path=str(netlist_path))
        circuit = parser.build_circuit()
        circuit.include('../libraries/ptm_45nm.txt')

        sl_array= (np.array(img1)-0.5)*2
        support_data= -(np.array(img2)*2)+1.6
        print('slarray',sl_array.shape)
        print('support_data',support_data.shape)

        #assert sl_array.shape[0] == 100
        #assert support_data.shape[0] == 100
        #print('slarray',sl_array.shape)
        #print('support_data',support_data.shape)
        
        for	waynum in range(0,64):
            ml='ML'+str(waynum)
            sl_tmp=sl_array[waynum]
            sup_tmp=support_data[waynum]

            for cellnum in range(0,100):            
                sl_vth='sl_way'+str(waynum)+'_cell'+str(cellnum)
                gate_node='gate_way'+str(waynum)+'_cell'+str(cellnum)
                inter='inter_way'+str(waynum)+'_cell'+str(cellnum)
                vgate='vgate_way'+str(waynum)+'_cell'+str(cellnum)

                circuit.V(sl_vth,gate_node,inter, str(sl_tmp[cellnum]))
                circuit.V(vgate,inter,'GND', str(sup_tmp[cellnum]))
            circuit.V(ml,ml, 'GND', '1.2')
        #simulator = circuit.simulator()
        simulator = circuit.simulator(temperature=25, nominal_temperature=25,simulator='ngspice-subprocess')
        analysis = simulator.operating_point()
        dist_current=[(float(-analysis.VML0)),(float(-analysis.VML1)),(float(-analysis.VML2)),(float(-analysis.VML3)),(float(-analysis.VML4)),(float(-analysis.VML5)),(float(-analysis.VML6)),(float(-analysis.VML7)),(float(-analysis.VML8)),(float(-analysis.VML9)),(float(-analysis.VML10)),(float(-analysis.VML11)),(float(-analysis.VML12)),(float(-analysis.VML13)),(float(-analysis.VML14)),(float(-analysis.VML15)),(float(-analysis.VML16)),(float(-analysis.VML17)),(float(-analysis.VML18)),(float(-analysis.VML19)),(float(-analysis.VML20)),(float(-analysis.VML21)),(float(-analysis.VML22)),(float(-analysis.VML23)),(float(-analysis.VML24)),(float(-analysis.VML25)),(float(-analysis.VML26)),(float(-analysis.VML27)),(float(-analysis.VML28)),(float(-analysis.VML29)),(float(-analysis.VML30)),(float(-analysis.VML31)),(float(-analysis.VML32)),(float(-analysis.VML33)),(float(-analysis.VML34)),(float(-analysis.VML35)),(float(-analysis.VML36)),(float(-analysis.VML37)),(float(-analysis.VML38)),(float(-analysis.VML39)),(float(-analysis.VML40)),(float(-analysis.VML41)),(float(-analysis.VML42)),(float(-analysis.VML43)),(float(-analysis.VML44)),(float(-analysis.VML45)),(float(-analysis.VML46)),(float(-analysis.VML47)),(float(-analysis.VML48)),(float(-analysis.VML49)),(float(-analysis.VML50)),(float(-analysis.VML51)),(float(-analysis.VML52)),(float(-analysis.VML53)),(float(-analysis.VML54)),(float(-analysis.VML55)),(float(-analysis.VML56)),(float(-analysis.VML57)),(float(-analysis.VML58)),(float(-analysis.VML59)),(float(-analysis.VML60)),(float(-analysis.VML61)),(float(-analysis.VML62)),(float(-analysis.VML63))]
        #print('training dist current:',dist_current.shape)
        return dist_current

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

        sim_mem_dist=tf.py_function(func=self.distance_circuit, inp=[encoded_image_1,encoded_image_2],Tout=tf.float32)
        print('sim mem size',sim_mem_dist)
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=sim_mem_dist)

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

            #pred0,pred1=self.model_train.predict_on_batch(images)
            #print('pred0:',pred0)
            #print('pred1:',pred1)
            #print('sub',100*(pred0-pred1))

            #train_loss, train_accuracy = self.model_train.train_on_batch(images, labels)
            train_loss = self.model_train.train_on_batch(images, labels)
            
            #pred=self.model_test.predict_on_batch(images)
            #print('pred',pred)
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
                evaluation_accuracy = self.omniglot_loader.one_shot_test_Sim_Mem(
                    self.model_test, support_set_size, 40, is_validation=False)
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
        return best_validation_accuracy,eval_accuracies
