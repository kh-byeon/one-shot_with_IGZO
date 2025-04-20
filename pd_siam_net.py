# -*- coding: utf-8 -*-
import os
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
from t_omniglot_loader import OmniglotLoader
import pandas as pd
import csv



class PulseBasedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, Pmax=200, maxNumLevelLTP=100, maxNumLevelLTD=100,
                 NL_LTP=0.1, NL_LTD=-0.1, Gmax=0.3, Gmin=-0.3, **kwargs):
        super(PulseBasedConv2D, self).__init__(filters, kernel_size, **kwargs)
        
        self.Pmax = Pmax  
        self.maxNumLevelLTP = maxNumLevelLTP
        self.maxNumLevelLTD = maxNumLevelLTD
        self.NL_LTP = NL_LTP
        self.NL_LTD = NL_LTD
        self.Gmax = Gmax
        self.Gmin = Gmin

        self.B_LTP = None
        self.B_LTD = None
        self.vth_shifts = None
    def build(self, input_shape):
        super(PulseBasedConv2D, self).build(input_shape)
        
        self.P = self.add_weight(
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Constant(self.Pmax / 2),
            trainable=False,
            name="Pulse"
        )
        
        self.A_LTP = self.get_paramA(self.NL_LTP, self.maxNumLevelLTP)
        self.A_LTD = self.get_paramA(self.NL_LTD, self.maxNumLevelLTD) 

        self.B_LTP = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTP / self.A_LTP))
        self.B_LTD = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTD / self.A_LTD))  

    @staticmethod
    def get_paramA(NL, numLevel):
        return (1 / abs(NL)) * numLevel
    
    @staticmethod
    def InvNonlinearWeight(conductance, A, B, minConductance):
        return -A * tf.math.log(1 - (conductance - minConductance) / B)

    @staticmethod
    def NonlinearWeight(xPulse, A, B, minConductance):
        return B * (1 - tf.math.exp(-xPulse / A)) + minConductance

    @tf.function
    def pulse_update(self, gradient,alpha):
        #alpha = 0.009  # Scaling factor for gradient step size

        if gradient is None:
            return

        if isinstance(gradient, list):
            gradient = tf.convert_to_tensor(gradient)  

        if gradient.shape != self.P.shape:
            #print(f"Shape Mismatch! P: {self.P.shape}, Gradient: {gradient.shape}")
            return  

        delta_P = alpha * (gradient / (tf.reduce_max(tf.abs(gradient), axis=-1, keepdims=True) + 1e-6))

        new_P = tf.clip_by_value(self.P + delta_P, -self.Pmax, self.Pmax)
        new_P = tf.round(new_P)  
        #tf.print('new_P',new_P)
        new_P = tf.where(tf.math.is_nan(new_P), tf.ones_like(new_P) * self.Pmax / 2, new_P)
        old_xPulse = self.InvNonlinearWeight(self.kernel, self.A_LTP, self.B_LTP, self.Gmin)
        #tf.print('old_xPulse',old_xPulse)
        xPulse = old_xPulse + delta_P

        new_weight_LTD = self.NonlinearWeight(xPulse, self.A_LTD, self.B_LTD, self.Gmax)
        new_weight_LTP = self.NonlinearWeight(xPulse, self.A_LTP, self.B_LTP, self.Gmin)

        new_weight = tf.where(delta_P >= 0, new_weight_LTP, new_weight_LTD)

        new_weight = tf.clip_by_value(new_weight, self.Gmin + 0.01, self.Gmax - 0.01)
        #tf.print('new_weight', new_weight)
    
        self.P.assign(new_P)
        self.kernel.assign(new_weight)


    def call(self, inputs, training=False):
        return super(PulseBasedConv2D, self).call(inputs)


class PulseBasedDense(tf.keras.layers.Dense):
    def __init__(self, units, Pmax=200, maxNumLevelLTP=100, maxNumLevelLTD=100,
                 NL_LTP=1, NL_LTD=-1, Gmax=0.3, Gmin=-0.3, **kwargs):
        super(PulseBasedDense, self).__init__(units, **kwargs)
        
        self.Pmax = Pmax  
        self.maxNumLevelLTP = maxNumLevelLTP
        self.maxNumLevelLTD = maxNumLevelLTD
        self.NL_LTP = NL_LTP
        self.NL_LTD = NL_LTD
        self.Gmax = Gmax
        self.Gmin = Gmin

        self.B_LTP = None
        self.B_LTD = None

    def build(self, input_shape):
        super(PulseBasedDense, self).build(input_shape)
        
        self.P = self.add_weight(
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Constant(self.Pmax / 2),
            trainable=False,
            name="Pulse"
        )
        
        self.A_LTP = self.get_paramA(self.NL_LTP, self.maxNumLevelLTP)
        self.A_LTD = self.get_paramA(self.NL_LTD, self.maxNumLevelLTD) 

        self.B_LTP = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTP / self.A_LTP))
        self.B_LTD = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTD / self.A_LTD))  

    @staticmethod
    def get_paramA(NL, numLevel):
        return (1 / abs(NL)) * numLevel
    
    @staticmethod
    def InvNonlinearWeight(conductance, A, B, minConductance):
        return -A * tf.math.log(1 - (conductance - minConductance) / B)

    @staticmethod
    def NonlinearWeight(xPulse, A, B, minConductance):
        return B * (1 - tf.math.exp(-xPulse / A)) + minConductance

    @tf.function
    def pulse_update(self, gradient,alpha):
        #alpha = 0.009  # Scaling factor for gradient step size

        if gradient is None:
            return

        if isinstance(gradient, list):
            gradient = tf.convert_to_tensor(gradient)  

        if gradient.shape != self.P.shape:
            #print(f"Shape Mismatch! P: {self.P.shape}, Gradient: {gradient.shape}")
            return  
        
        delta_P = alpha * (gradient / (tf.reduce_max(tf.abs(gradient), axis=-1, keepdims=True) + 1e-6))

        new_P = tf.clip_by_value(self.P + delta_P, -self.Pmax, self.Pmax)
        new_P = tf.round(new_P)  
        #tf.print('new_P',new_P)
        new_P = tf.where(tf.math.is_nan(new_P), tf.ones_like(new_P) * self.Pmax / 2, new_P)
        old_xPulse = self.InvNonlinearWeight(self.kernel, self.A_LTP, self.B_LTP, self.Gmin)
        xPulse = old_xPulse + delta_P
        
        new_weight_LTD = self.NonlinearWeight(xPulse, self.A_LTD, -self.B_LTD, self.Gmax)
        new_weight_LTP = self.NonlinearWeight(xPulse, self.A_LTP, self.B_LTP, self.Gmin)

        new_weight = tf.where(delta_P >= 0, new_weight_LTP, new_weight_LTD)

        new_weight = tf.clip_by_value(new_weight, self.Gmin + 0.01, self.Gmax - 0.01)
    
        self.P.assign(new_P)
        self.kernel.assign(new_weight)


    def call(self, inputs, training=False):
        return super(PulseBasedDense, self).call(inputs)



class SiameseNetwork:
    def __init__(self, dataset_path, learning_rate, batch_size, use_augmentation,
                 learning_rate_multipliers, l2_regularization_penalization, margin):
        self.input_shape = (105, 105, 1)  
        self.model_train = []
        self.model_test = []
        self.learning_rate = learning_rate
        self.margin = margin
        self.batch_size = batch_size
        self.omniglot_loader = OmniglotLoader(
            dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        self._construct_siamese_architecture(learning_rate_multipliers,
                                             l2_regularization_penalization)
        self.contrastive_loss_w_margin(margin)

    def contrastive_loss_w_margin(self, margin):
        def contrastive_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(tf.cast(margin, tf.float32) - y_pred, 0))
            return (1 / 2) * (y_true * square_pred + (1 - y_true) * margin_square)
        return contrastive_loss

    def _construct_siamese_architecture(self, learning_rate_multipliers, l2_regularization_penalization):
        if not hasattr(self, 'vth_shifts') or self.vth_shifts is None:
            # Create Vth shifts for 100 output vectors with sigma = 0.038
            self.vth_shifts = np.random.normal(0, 0.038, 100).astype(np.float32)  # 100 shifts for 100 output vectors

        convolutional_net = Sequential()
        """
        # 1️첫 번째 Conv Layer만 PulseBasedConv2D 사용
        convolutional_net.add(PulseBasedConv2D(
            filters=64,
            kernel_size=(13, 13),
            activation='relu',
            kernel_regularizer=l2(l2_regularization_penalization['Conv1']),
            name='Conv1'
        ))
        convolutional_net.add(MaxPool2D())
        
        # 2️나머지 Conv Layer들은 일반 keras.layers.Conv2D 사용
        for i, kernel_size in zip(range(2, 5), [(11, 11), (7, 7), (5, 5)]):  
            convolutional_net.add(tf.keras.layers.Conv2D(
                filters=64 if i < 4 else 128,  
                kernel_size=kernel_size,
                activation='relu',
                kernel_regularizer=l2(l2_regularization_penalization[f'Conv{i}']),
                name=f'Conv{i}'
            ))
            convolutional_net.add(MaxPool2D())

        convolutional_net.add(Flatten())
        """
        for i, kernel_size in zip(range(1, 5), [(13,13), (11, 11), (7, 7), (5, 5)]):  
            convolutional_net.add(tf.keras.layers.Conv2D(
                filters=64 if i < 4 else 128,  
                kernel_size=kernel_size,
                activation='relu',
                kernel_regularizer=l2(l2_regularization_penalization[f'Conv{i}']),
                name=f'Conv{i}'
            ))
            convolutional_net.add(MaxPool2D())

        convolutional_net.add(Flatten())
        # 3️Dense Layer는 기존대로 PulseBasedDense 사용
        convolutional_net.add(PulseBasedDense(
            units=1000, activation='tanh',
            kernel_regularizer=l2(l2_regularization_penalization['Dense1']),
            name='Dense1'
        ))
        convolutional_net.add(PulseBasedDense(
            units=100, activation='tanh',
            name='Dense3'
        ))
        """
        convolutional_net.add(tf.keras.layers.Dense(
            units=1000, activation='tanh',
            kernel_regularizer=l2(l2_regularization_penalization['Dense1']),
            name='Dense1'
        ))
        convolutional_net.add(tf.keras.layers.Dense(
            units=100, activation='tanh',
            name='Dense3'
        ))
        """
        # Siamese Network의 두 개의 입력
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        # 두 입력 이미지를 같은 네트워크에 통과시킴
        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)
        """
        # L2 Distance Layer 생성
        l2_distance_layer = Lambda(lambda tensors: tf.expand_dims(
            K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1)), axis=-1))
        l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=l2_distance)
        self.model_test = Model(inputs=input_image_1, outputs=encoded_image_1)
        """
        vth_shifts_tensor = tf.constant(self.vth_shifts, dtype=tf.float32)
        coef = 1.0
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
        
        sim_mem_distance = sim_mem_distance_layer([encoded_image_1 * 1.5, encoded_image_2 * 1.5])
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=sim_mem_distance * 100000)
        self.model_test = Model(inputs=input_image_1, outputs=encoded_image_1)
       


    def apply_sgd_update(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model_train(images, training=True)

            
            if tf.math.is_nan(tf.reduce_sum(predictions)):
                raise ValueError("Predictions contain NaN! Check model output!")

            loss = self.contrastive_loss_w_margin(self.margin)(labels, predictions)
            loss = tf.reduce_mean(loss)

            
            if tf.math.is_nan(loss):
                raise ValueError("Loss is NaN! Check loss function and predictions!")

        # 3. Gradient 계산
        gradients = tape.gradient(loss, self.model_train.trainable_variables)

        # 4. Gradient NaN 확인
        gradient_nan_check = [tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.float32)) for g in gradients if g is not None]
    
        if any(tf.math.is_nan(tf.reduce_sum(g)) for g in gradients if g is not None):
            raise ValueError("Gradient contains NaN!")

        # 5. Gradient Clipping (NaN 방지)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients if g is not None]

        self.optimizer.apply_gradients(zip(gradients, self.model_train.trainable_variables))

        return gradients, loss


    def train_step(self, gradients, optimizer):
        # Conv2D 레이어의 학습 가능한 변수 가져오기 (Conv1~Conv3은 SGD, Conv4만 Pulse Update)
        conv2d_vars = [var for layer in self.model_train.layers if isinstance(layer, Sequential)
                       for sub_layer in layer.layers if isinstance(sub_layer, tf.keras.layers.Conv2D)
                       for var in sub_layer.trainable_variables]

        dense_vars = [var for layer in self.model_train.layers if isinstance(layer, Sequential)
                      for sub_layer in layer.layers if isinstance(sub_layer, tf.keras.layers.Dense)
                      for var in sub_layer.trainable_variables]

        conv2d_var_refs = [v.ref() for v in conv2d_vars]  # Conv2D 변수 참조 리스트
        dense_var_refs = [v.ref() for v in dense_vars]  # Dense 변수 참조 리스트

        # Conv1~Conv4 SGD로 업데이트
        optimizer.apply_gradients(zip([grad for grad, var in zip(gradients, self.model_train.trainable_variables)
                                       if var.ref() in conv2d_var_refs or var.ref() in dense_var_refs], 
                                      conv2d_vars + dense_vars))

        #pulse update는 iterator에서



    def train_siamese_network(self, number_of_iterations, support_set_size,
                          validate_each, evaluate_each, model_name):

        self.omniglot_loader.split_train_datasets()
    
        # Optimizer를 클래스 변수로 선언하여 여러 번 생성되지 않도록 수정
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        best_accuracy_iteration = 0
        best_validation_accuracy = 0.0
        val_count, eval_count = 0, 0
        valid_accuracies = np.zeros(shape=(number_of_iterations // validate_each))
        eval_accuracies = np.zeros(shape=(number_of_iterations // evaluate_each))
        initial_alpha = 0.1
        alpha = initial_alpha
        loss_log_file = 'loss_log.csv'

        # **Loss 기록 파일 초기화**
        if not os.path.exists(loss_log_file):
            with open(loss_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Loss"])  # 헤더 추가
        for iteration in range(number_of_iterations):
            images, labels = self.omniglot_loader.get_train_batch()
            gradients, loss = self.apply_sgd_update(images, labels)
            self.train_step(gradients,self.optimizer)

            for layer in self.model_train.layers:
                if isinstance(layer, Sequential):
                    for sub_layer in layer.layers:
                        if isinstance(sub_layer, PulseBasedConv2D) or isinstance(sub_layer, PulseBasedDense):
                            # **gradient와 kernel의 shape이 맞을 경우에만 pulse_update 적용**
                            for grad, var in zip(gradients, self.model_train.trainable_variables):
                                if var.ref() == sub_layer.kernel.ref():
                                    sub_layer.pulse_update(grad, alpha)

            # **3. Alpha 값 감소 (500번마다 0.95씩 감소)**
            if (iteration + 1) % 500 == 0:
                alpha *= 1.0  # 500 itera

            current_lr = K.get_value(self.learning_rate)
            print(f"Iteration {iteration}/{number_of_iterations} - Train Loss: {loss.numpy():.6f}, LR: {current_lr:.3E}, Alpha = {alpha:.6f}")

            if (iteration + 1) % 500 == 0:
                self.learning_rate = self.learning_rate * 0.99

            if (iteration + 1) % validate_each == 0:
                validation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model_test, support_set_size, 40, is_validation=True)

                valid_accuracies[val_count] = validation_accuracy
                val_count += 1

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration
                    self.model_test.save_weights(f'bestmodel/{model_name}.h5')
            if (iteration + 1) % 10 == 0:
                with open(loss_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration + 1, loss.numpy()])  # Loss 기록

            if (iteration +1) % evaluate_each == 0:
                evaluation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model_test, support_set_size, 40, is_validation=False)
                eval_accuracies[eval_count]=evaluation_accuracy
                eval_count += 1
                with open('accuracy_by_vthshift.csv','a') as c:
                    wr=csv.writer(c)
                    wr.writerow(eval_accuracies)

            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' + str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Training Completed!')
        return best_validation_accuracy, eval_accuracies


