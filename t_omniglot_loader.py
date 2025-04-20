import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from pathlib import Path
import math
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
import warnings
from PySpice.Spice.Xyce.Server import XyceServer
import random
import numpy as np
import math
from PIL import Image
import tensorflow.keras.backend as K
import tensorflow as tf
from image_augmentor import ImageAugmentor
logger.setLevel(logging.ERROR)
import csv
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
class OmniglotLoader:
    """Class that loads and prepares the Omniglot dataset

    This Class was constructed to read the Omniglot alphabets, separate the 
    training, validation and evaluation test. It also provides function for
    geting one-shot task batches.

    Attributes:
        dataset_path: path of Omniglot Dataset
        train_dictionary: dictionary of the files of the train set (background set). 
            This dictionary is used to load the batch for training and validation.
        evaluation_dictionary: dictionary of the evaluation set. 
        image_width: self explanatory
        image_height: self explanatory
        batch_size: size of the batch to be used in training
        use_augmentation: boolean that allows us to select if data augmentation is 
            used or not
        image_augmentor: instance of class ImageAugmentor that augments the images
            with the affine transformations referred in the paper

    """

    def __init__(self, dataset_path, use_augmentation, batch_size):
        """Inits OmniglotLoader with the provided values for the attributes.

        It also creates an Image Augmentor object and loads the train set and 
        evaluation set into dictionaries for future batch loading.

        Arguments:
            dataset_path: path of Omniglot dataset
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not       
            batch_size: size of the batch to be used in training     
        """

        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 105
        self.image_height = 105
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._train_alphabets = []
        self._validation_alphabets = []
        self._evaluation_alphabets = []
        self._current_train_alphabet_index = 0
        self._current_validation_alphabet_index = 0
        self._current_evaluation_alphabet_index = 0

        self.load_dataset()
        # Vth shift data (to be initialized later)
        self.vth_shifts = None
        self.shifted_die_data = None  # Shifted data for 3rd die
        if (self.use_augmentation):
            self.image_augmentor = self.createAugmentor()
        else:
            self.use_augmentation = []

    def load_dataset(self):
        """Loads the alphabets into dictionaries

        Loads the Omniglot dataset and stores the available images for each
        alphabet for each of the train and evaluation set.

        """

        train_path = os.path.join(self.dataset_path, 'images_background')
        validation_path = os.path.join(self.dataset_path, 'images_evaluation')

        # First let's take care of the train alphabets
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)

            current_alphabet_dictionary = {}

            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)

                current_alphabet_dictionary[character] = os.listdir(
                    character_path)

            self.train_dictionary[alphabet] = current_alphabet_dictionary

        # Now it's time for the validation alphabets
        for alphabet in os.listdir(validation_path):
            alphabet_path = os.path.join(validation_path, alphabet)

            current_alphabet_dictionary = {}

            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)

                current_alphabet_dictionary[character] = os.listdir(
                    character_path)

            self.evaluation_dictionary[alphabet] = current_alphabet_dictionary

    def createAugmentor(self):
        """ Creates ImageAugmentor object with the parameters for image augmentation

        Rotation range was set in -15 to 15 degrees
        Shear Range was set in between -0.3 and 0.3 radians
        Zoom range between 0.8 and 2 
        Shift range was set in +/- 5 pixels

        Returns:
            ImageAugmentor object

        """
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]

        return ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)

    


    def split_train_datasets(self):
        """ Splits the train set in train and validation

        Divide the 30 train alphabets in train and validation with
        # a 80% - 20% split (24 vs 6 alphabets)

        """

        available_alphabets = list(self.train_dictionary.keys())
        number_of_alphabets = len(available_alphabets)

        train_indexes = random.sample(
            range(0, number_of_alphabets - 1), int(0.8 * number_of_alphabets))

        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self._train_alphabets.append(available_alphabets[index])
            available_alphabets.pop(index)

        # The remaining alphabets are saved for validation
        self._validation_alphabets = available_alphabets
        self._evaluation_alphabets = list(self.evaluation_dictionary.keys())

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            (number_of_pairs, self.image_height, self.image_height, 1)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2])
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()

            pairs_of_images[0][pair, :, :, 0] = image
            image = Image.open(path_list[pair * 2 + 1])
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()

            pairs_of_images[1][pair, :, :, 0] = image
            if not is_one_shot_task:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_images[0][:, :, :,
                               :] = pairs_of_images[0][random_permutation, :, :, :]
            pairs_of_images[1][:, :, :,
                               :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

    def test_convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        #print('path list',path_list)

        number_of_pairs = int(len(path_list))

        pairs_of_images = [np.zeros(
            (number_of_pairs, self.image_height, self.image_height, 1)) for i in range(1)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair])
            image = np.asarray(image).astype(np.float64)
            
            #print('img',image)
            image = image / image.std() - image.mean()
            
            #print('centerd-img',image)
            pairs_of_images[0][pair, :, :, 0] = image
        

        return pairs_of_images, labels


    def get_train_batch(self):
        """ Loads and returns a batch of train images

        Get a batch of pairs from the training set. Each batch will contain
        images from a single alphabet. I decided to select one single example
        from random n/2 characters in each alphabet. If the current alphabet
        has lower number of characters than n/2 (some of them have 14) we
        sample repeated classed for that batch per character in the alphabet
        to pair with a different categories. In the other half of the batch
        I selected pairs of same characters. In resume we will have a batch
        size of n, with n/2 pairs of different classes and n/2 pairs of the same
        class. Each batch will only contains samples from one single alphabet.

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes
        label.shape = (batch_size*2,1)
        image.shape = (2,batch_size*2,105,105,1)

        """

        current_alphabet = self._train_alphabets[self._current_train_alphabet_index]
        #print('tot alpha',current_alphabet[:])
        available_characters = list(
            self.train_dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        # If the number of classes if less than self.batch_size/2
        # we have to repeat characters
        selected_characters_indexes = [random.randint(
            0, number_of_characters-1) for i in range(self.batch_size)]   #batch= 32
        
        for index in selected_characters_indexes:
            current_character = available_characters[index]
            available_images = (self.train_dictionary[current_alphabet])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, 'images_background', current_alphabet, current_character)

            # Random select a 3 indexes of images from the same character (Remember
            # that for each character we have 20 examples).
            image_indexes = random.sample(range(0, 20), 3)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(image)
            image = os.path.join(
                image_path, available_images[image_indexes[1]])
            bacth_images_path.append(image)

            # Now let's take care of the pair of images from different characters
            image = os.path.join(
                image_path, available_images[image_indexes[2]])
            bacth_images_path.append(image)
            different_characters = available_characters[:]
            different_characters.pop(index)
            different_character_index = random.sample(
                range(0, number_of_characters - 1), 1)
            current_character = different_characters[different_character_index[0]]
            available_images = (self.train_dictionary[current_alphabet])[
                current_character]
            image_indexes = random.sample(range(0, 20), 1)
            image_path = os.path.join(
                self.dataset_path, 'images_background', current_alphabet, current_character)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(image)

        self._current_train_alphabet_index += 1

        if (self._current_train_alphabet_index > 23):
            self._current_train_alphabet_index = 0
        #print('img path',bacth_images_path)
        images, labels = self._convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation:
            images = self.image_augmentor.get_random_transform(images)

        return images, labels

    def mod_get_train_batch(self):
        """ Loads and returns a batch of train images

        Get a batch of pairs from the training set. Each batch will contain
        images from a single alphabet. I decided to select one single example
        from random n/2 characters in each alphabet. If the current alphabet
        has lower number of characters than n/2 (some of them have 14) we
        sample repeated classed for that batch per character in the alphabet
        to pair with a different categories. In the other half of the batch
        I selected pairs of same characters. In resume we will have a batch
        size of n, with n/2 pairs of different classes and n/2 pairs of the same
        class. Each batch will only contains samples from one single alphabet.

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes
        label.shape = (batch_size*2,1)
        image.shape = (2,batch_size*2,105,105,1)
        
        for label modified for different alphabet train set
        """
        #self._current_train_alphabet_index=0
        current_alphabet = self._train_alphabets[self._current_train_alphabet_index]  #1of80%of train alphabets
        #print('tot alpha',current_alphabet)
        available_characters = list(self.train_dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        # If the number of classes if less than self.batch_size/2
        # we have to repeat characters
        same_characters_indexes = [random.randint(0, number_of_characters-1) for i in range(int(self.batch_size*(1/4)))]#batch= 32
        dif_characters_indexes = [random.randint(0, number_of_characters-1) for i in range(int(self.batch_size*(3/4)))]#batch= 32
        
        for index in same_characters_indexes:   #8 same set from the same characters
            current_character = available_characters[index]
            available_images = (self.train_dictionary[current_alphabet])[current_character]
            image_path = os.path.join(
                self.dataset_path, 'images_background', current_alphabet, current_character)

            # Random select a 2 indexes of images from the same character (Remember
            # that for each character we have 20 examples).
            image_indexes = random.sample(range(0, 20), 2)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(image)
            image = os.path.join(
                image_path, available_images[image_indexes[1]])
            bacth_images_path.append(image)


        for index in dif_characters_indexes:    #24 dif characters
            # Now let's take care of the pair of images from different characters
            alphabet_indexes = random.sample(range(0, 23), 2)
            #current_alphabet = self._train_alphabets[alphabet_indexes[0]]
            current_alphabet = self._train_alphabets[self._current_train_alphabet_index]

            available_characters = list(self.train_dictionary[current_alphabet].keys())
            number_of_characters = len(available_characters)

            current_character = available_characters[random.randint(0,number_of_characters-1)]

            available_images = (self.train_dictionary[current_alphabet])[current_character]
            image_path = os.path.join(self.dataset_path, 'images_background', current_alphabet, current_character)
            image = os.path.join(image_path, available_images[random.randint(0, 19)])
            bacth_images_path.append(image)


            different_alphabet = self._train_alphabets[alphabet_indexes[1]]
            available_characters = list(self.train_dictionary[different_alphabet].keys())
            number_of_characters = len(available_characters)
            current_character = available_characters[random.randint(0,number_of_characters-1)]
            available_images = (self.train_dictionary[different_alphabet])[current_character]
            image_path = os.path.join(self.dataset_path, 'images_background', different_alphabet, current_character)
            image = os.path.join(image_path, available_images[random.randint(0, 19)])
            bacth_images_path.append(image)
        #print('img path',bacth_images_path)
            

        self._current_train_alphabet_index += 1

        if (self._current_train_alphabet_index > 23):
            self._current_train_alphabet_index = 0

        images, dmmy = self._convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation:
            images = self.image_augmentor.get_random_transform(images)
        ones=np.ones((8,1))
        zeros=np.zeros((24, 1))
        labels = np.concatenate((ones,zeros),axis=0)
        #print('label',labels)
        return images, labels


    def get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images

        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            alphabets = self._validation_alphabets
            current_alphabet_index = self._current_validation_alphabet_index
            image_folder_name = 'images_background'
            dictionary = self.train_dictionary
        else:
            alphabets = self._evaluation_alphabets
            current_alphabet_index = self._current_evaluation_alphabet_index
            image_folder_name = 'images_evaluation'
            dictionary = self.evaluation_dictionary

        current_alphabet = alphabets[current_alphabet_index]
        available_characters = list(dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        test_character_index = random.sample(
            range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]

        available_images = (dictionary[current_alphabet])[current_character]

        image_indexes = random.sample(range(0, 20), 2)
        image_path = os.path.join(
            self.dataset_path, image_folder_name, current_alphabet, current_character)

        test_image = os.path.join(
            image_path, available_images[image_indexes[0]])
        bacth_images_path.append(test_image)
        image = os.path.join(
            image_path, available_images[image_indexes[1]])
        bacth_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = (dictionary[current_alphabet])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, current_alphabet, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(test_image)
            bacth_images_path.append(image)
            #print('\n BATCH IMAGE',bacth_images_path)
        images, labels = self._convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=True)

        return images, labels


    def mod_get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images
            Gets a one-shot batch for evaluation or validation set, it consists in a
            single image that will be compared with a support set of images. It returns
            the pair of images to be compared by the model and it's labels (the first
            pair is always 1) and the remaining ones are 0's
            Returns:
                pairs_of_images: pairs of images for the current batch
                labels: correspondent labels -1 for same class, 0 for different classes
        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            alphabets = self._validation_alphabets
            current_alphabet_index = self._current_validation_alphabet_index
            image_folder_name = 'images_background'
            dictionary = self.train_dictionary
        else:
            alphabets = self._evaluation_alphabets
            current_alphabet_index = self._current_evaluation_alphabet_index
            image_folder_name = 'images_evaluation'
            dictionary = self.evaluation_dictionary

        current_alphabet = alphabets[current_alphabet_index]
        available_characters = list(dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        #print('cur alpha',current_alphabet)
        #print('tot alpha',alphabets[:])   it's fixed list

        bacth_images_path = []

        test_character_index = random.sample(range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]
        
        q_available_images = (dictionary[current_alphabet])[current_character]

        q_image_indexes = random.sample(range(0, 20), 2)
        q_image_path = os.path.join(
            self.dataset_path, image_folder_name, current_alphabet, current_character)

        test_image = os.path.join(
            q_image_path, q_available_images[q_image_indexes[0]])
        bacth_images_path.append(test_image)        


        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)
        support_alphabet=alphabets[:]
        #print('support_alphabet1',len(support_alphabet))
        support_alphabet.pop(self._current_validation_alphabet_index)
        #print('support_alphabet len',len(support_alphabet))
        dif_alpha_index=random.sample(range(0,len(support_alphabet)),4)
        #print('index',dif_alpha_index)
        tmp_cnt=0
        for index in support_characters_indexes:
            dif_alphabet = support_alphabet[dif_alpha_index[tmp_cnt]]
            #print('dif_alpha',dif_alphabet)
            available_characters = list(dictionary[dif_alphabet].keys())
            #print('available_characters',available_characters)
            current_character = available_characters[random.randint(0,len(available_characters)-1)]

            available_images = (dictionary[dif_alphabet])[current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, dif_alphabet, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            #bacth_images_path.append(test_image)
            bacth_images_path.append(image)
            #print('\n BATCH IMAGE',bacth_images_path)
            #print('here')
            tmp_cnt +=1
        #print('q_available_images',q_available_images)
        q_image = os.path.join(
            q_image_path, q_available_images[q_image_indexes[1]])
        bacth_images_path.append(q_image)
        #print('bacth_images_path',bacth_images_path)

        images, labels = self.test_convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_alphabet,
                      is_validation):
        """ Prepare one-shot task and evaluate its performance
        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet
        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """
        print("\nOne-shot test with Euclidean distance")
        # Set some variables that depend on dataset
        if is_validation:
            alphabets = self._validation_alphabets
            print('\nMaking One Shot Task on validation alphabets:')
        else:
            alphabets = self._evaluation_alphabets
            print('\nMaking One Shot Task on evaluation alphabets:')

        mean_global_accuracy = 0

        for alphabet in alphabets:
            mean_alphabet_accuracy = 0
            for _ in range(number_of_tasks_per_alphabet): # 40
                images, _ = self.mod_get_one_shot_batch(support_set_size, is_validation=is_validation)
                #images, _ = self.get_one_shot_batch(support_set_size, is_validation=is_validation)
                #print("image",images)
                prediction1 = model.predict_on_batch(images)
                #with open('emb_output.csv','a') as c:
                    #wr=csv.writer(c)
                    #wr.writerow(prediction1)
                #tmp=np.array(prediction1)
                #print('predict',prediction1)
                dist_btw_embbeding=[]
                for sup in range(len(prediction1)-1):
                    dist_btw_embbeding.append(tf.expand_dims(K.sqrt(K.sum(K.square(prediction1[sup]-prediction1[5]),axis=-1)),axis=-1))
                #print('prob',probabilities)
                #print('std',probabilities.std())
                # Added this condition because noticed that sometimes the outputs
                # of the classifier was almost the same in all images, meaning that
                # the argmax would be always by defenition 0.
                #print('dist',dist_btw_embbeding)

                if np.argmin(dist_btw_embbeding) == 0:
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                mean_alphabet_accuracy += accuracy
                mean_global_accuracy += accuracy

            mean_alphabet_accuracy /= number_of_tasks_per_alphabet

            print(alphabet + ' alphabet' + ', accuracy: ' +
                  str(mean_alphabet_accuracy))
            if is_validation:
                self._current_validation_alphabet_index += 1
            else:
                self._current_evaluation_alphabet_index += 1

        mean_global_accuracy /= (len(alphabets) *
                                 number_of_tasks_per_alphabet)

        print('\nMean global accuracy: ' + str(mean_global_accuracy))

        # reset counter
        if is_validation:
            self._current_validation_alphabet_index = 0
        else:
            self._current_evaluation_alphabet_index = 0

        return mean_global_accuracy

    
    def initialize_vth_shifted_data(self, csv_path, die_index=2, num_devices_per_die=10, num_points_per_device=121, num_output_vectors=100, sigma=0.038):
        """
        Initialize Vth shifted data based on die data read from CSV with symmetric reflection for VG
        and generate 100 U-shaped IV curves by combining two shifted curves for CNN output.
        """
        
        file_names = [f"wf7_{i}.csv" for i in range(1, 11)]
        all_data = []

        
        for file_name in file_names:
            file_path = f"{csv_path}/{file_name}" 
            data = pd.read_csv(file_path)

            
            vd_data = data[data['VD'] == 0.1][['VG', 'ID']].copy()
            all_data.append(vd_data)

        
        combined_data = pd.concat(all_data, ignore_index=True)

        
        die_data = combined_data.iloc[die_index * num_devices_per_die * num_points_per_device:
                                      (die_index + 1) * num_devices_per_die * num_points_per_device]

        
        self.vth_shifts = np.random.normal(0, sigma, num_output_vectors * 2)

        
        self.shifted_die_data = []
        for i in range(num_output_vectors):
            
            vth_shift_1 = self.vth_shifts[2 * i]
            vth_shift_2 = self.vth_shifts[2 * i + 1]

           
            start_idx_1 = (i % num_devices_per_die) * num_points_per_device
            end_idx_1 = start_idx_1 + num_points_per_device
            device_data_1 = die_data.iloc[start_idx_1:end_idx_1]

            start_idx_2 = ((i + 1) % num_devices_per_die) * num_points_per_device
            end_idx_2 = start_idx_2 + num_points_per_device
            device_data_2 = die_data.iloc[start_idx_2:end_idx_2]

            
            VG_shifted_1 = device_data_1['VG'].values + vth_shift_1
            ID_1 = device_data_1['ID'].values
            VG_shifted_2 = device_data_2['VG'].values + vth_shift_2
            ID_2 = device_data_2['ID'].values

            
            current_threshold = 1e-10
            valid_indices_1 = np.abs(ID_1) > current_threshold
            valid_indices_2 = np.abs(ID_2) > current_threshold
            VG_shifted_1 = VG_shifted_1[valid_indices_1]
            ID_1 = ID_1[valid_indices_1]
            VG_shifted_2 = VG_shifted_2[valid_indices_2]
            ID_2 = ID_2[valid_indices_2]

            
            VG_negative = -VG_shifted_1
            ID_negative = ID_1
            VG_positive = VG_shifted_2
            ID_positive = ID_2

            merged_VG = np.concatenate([VG_negative, VG_positive])
            merged_ID = np.concatenate([ID_negative, ID_positive])

            
            self.shifted_die_data.append({
                "VG": merged_VG,
                "ID": merged_ID
            })



    def one_shot_test_fitted_simmem(self, model, support_set_size, number_of_tasks_per_alphabet, is_validation):
        """
        Prepare one-shot task and evaluate its performance with custom distance calculation
        using Vth shifted VG/ID data and existing CNN outputs.
        """
        print('\nOne-shot task with sim-mem fitted circuit')
        if is_validation:
            alphabets = self._validation_alphabets
            print('\nMaking One Shot Task on validation alphabets:')
        else:
            alphabets = self._evaluation_alphabets
            print('\nMaking One Shot Task on evaluation alphabets:')

        mean_global_accuracy = 0
        coef = 1.0

        # Initialize Vth shifted data if not already initialized
        if not hasattr(self, 'vth_shifts') or self.vth_shifts is None:
            # Create Vth shifts for 100 output vectors with sigma = 0.038
            self.vth_shifts = np.random.normal(0, 0.038, 100).astype(np.float32)

        # Vth shifts as a TensorFlow constant
        vth_shifts_tensor = tf.constant(self.vth_shifts, dtype=tf.float32)

        for alphabet in alphabets:
            mean_alphabet_accuracy = 0
            for _ in range(number_of_tasks_per_alphabet):
                images, _ = self.mod_get_one_shot_batch(support_set_size, is_validation=is_validation)

                # Pass images through CNN to get query and support vectors
                prediction1 = model.predict_on_batch(images)
                query_data = prediction1[len(prediction1) - 1]  # Query vector (1 case)
                query_data *= 1.5
                support_data = prediction1[0:5]  # Support vectors (5 cases)
                support_data *= 1.5

                ml_current = []

                # Use Vth shifted distance calculation for each support vector
                for sup in support_data:
                    sim_mem_distance_layer = Lambda(lambda tensors: tf.expand_dims(
                        K.sum(
                            (4.718566e-24) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 11) +
                            (7.010791e-11) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 10) +
                            (-1.117418e-22) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 9) +
                            (-1.851356e-09) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 8) +
                            (9.202852e-22) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 7) +
                            (1.901891e-08) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 6) +
                            (-3.056228e-21) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 5) +
                            (-1.059278e-07) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 4) +
                            (3.414937e-21) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 3) +
                            (6.370100e-07) * tf.pow(coef * (tensors[0] - tensors[1] + vth_shifts_tensor), 2) +
                            (-6.888072e-22) * coef * (tensors[0] - tensors[1] + vth_shifts_tensor) +
                            (3.967141e-09), axis=-1
                        ), axis=-1))

                    sim_mem_distance = sim_mem_distance_layer([sup, query_data]) * 100000
                    ml_current.append(sim_mem_distance)

                # Find the closest vector
                if np.argmin(ml_current) == 0:
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                mean_alphabet_accuracy += accuracy
                mean_global_accuracy += accuracy

            mean_alphabet_accuracy /= number_of_tasks_per_alphabet

            print(f"{alphabet} alphabet, accuracy: {mean_alphabet_accuracy:.2f}")
            if is_validation:
                self._current_validation_alphabet_index += 1
            else:
                self._current_evaluation_alphabet_index += 1

        mean_global_accuracy /= (len(alphabets) * number_of_tasks_per_alphabet)

        print(f'\nMean global accuracy: {mean_global_accuracy:.2f}')

        # Reset counter
        if is_validation:
            self._current_validation_alphabet_index = 0
        else:
            self._current_evaluation_alphabet_index = 0

        return mean_global_accuracy
