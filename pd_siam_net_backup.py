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

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K



class PulseBasedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, Pmax=200, maxNumLevelLTP=100, maxNumLevelLTD=100,
                 NL_LTP=2, NL_LTD=-1, Gmax=0.5, Gmin=-0.5, **kwargs):
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
    def pulse_update(self, gradient):
        alpha = 0.1  # Scaling factor for gradient step size

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
                 NL_LTP=1, NL_LTD=-1, Gmax=0.1, Gmin=-0.1, **kwargs):
        super(PulseBasedDense, self).__init__(units, **kwargs)

        self.Pmax = Pmax  
        self.maxNumLevelLTP = maxNumLevelLTP
        self.maxNumLevelLTD = maxNumLevelLTD
        self.NL_LTP = NL_LTP
        self.NL_LTD = NL_LTD
        self.Gmax = Gmax
        self.Gmin = Gmin

        self.A_LTP = self.get_paramA(self.NL_LTP, self.maxNumLevelLTP)
        self.A_LTD = self.get_paramA(self.NL_LTD, self.maxNumLevelLTD) 

        self.B_LTP = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTP / self.A_LTP))
        self.B_LTD = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTD / self.A_LTD))

        # LTP, LTD 별도 LUT 생성
        self.LUT_LTP, self.LUT_LTD = self.create_lookup_table()

    def build(self, input_shape):
        super(PulseBasedDense, self).build(input_shape)

        # self.kernel을 안전하게 numpy 변환
        if isinstance(self.kernel, tf.Variable):
            initial_weight = self.kernel.read_value().numpy()
        else:
            initial_weight = self.kernel.numpy()

        # inv_lookup 적용 후 shape 검증
        initial_pulse = self.inv_lookup(initial_weight)

        assert initial_pulse.shape == self.kernel.shape, \
            f"Shape mismatch in build(): {initial_pulse.shape} vs {self.kernel.shape}"

        # `initial_pulse`가 NumPy일 경우, Tensor로 변환 후 `add_weight()`에 전달
        initial_pulse = tf.convert_to_tensor(initial_pulse, dtype=tf.float32)

        # `self.P` 생성 시 shape 유지 검증 후 추가
        self.P = self.add_weight(
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Constant(initial_pulse),
            trainable=False,
            name="Pulse"
        )

    @staticmethod
    def NonlinearWeight(xPulse, A, B, minConductance):
        return B * (1 - tf.math.exp(-xPulse / A)) + minConductance

    @staticmethod
    def get_paramA(NL, numLevel):
        return (1 / abs(NL)) * numLevel

    def create_lookup_table(self):
        P_values = np.arange(0, self.Pmax)  # 0 ~ Pmax까지 정수로

        # LTP용 LUT 생성
        B_LTP = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTP / self.A_LTP))
        LUT_LTP = self.NonlinearWeight(P_values[0:100], self.A_LTP, B_LTP, self.Gmin)

        # LTD용 LUT 생성
        B_LTD = (self.Gmax - self.Gmin) / (1 - np.exp(-self.maxNumLevelLTD / self.A_LTD))
        LUT_LTD = self.NonlinearWeight(P_values[100:200] - self.Pmax/2, self.A_LTD, -B_LTD, self.Gmax)
        tf.print('LUT_LTP min:',tf.reduce_min(LUT_LTP),'LUT_LTP max',tf.reduce_max(LUT_LTP))
        tf.print('LUT_LTD min:',tf.reduce_min(LUT_LTD),'LUT_LTD max',tf.reduce_max(LUT_LTD))
        return LUT_LTP, LUT_LTD

    def nonlinear_lookup(self, P):
        # TensorFlow 연산으로 변환 (NumPy 대신 사용)
        P = tf.round(tf.clip_by_value(P, 0, self.Pmax))
        P = tf.cast(P, tf.int32)  # 정수형 변환

        # LUT에서 값 조회 (tf.gather 사용)
        new_weight = tf.where(
            P < 100,
            tf.gather(self.LUT_LTP, P),
            tf.gather(self.LUT_LTD, P - 100)
        )
    
        return new_weight


    def inv_lookup(self, G):
        #tf.print('G min:', tf.reduce_min(G), 'G max:', tf.reduce_max(G))
        # 1. Weight 값 제한 (LUT 범위 벗어나지 않도록 조정)
        G = np.clip(G, np.min(self.LUT_LTP), np.max(self.LUT_LTP))
        tf.print('G min:', tf.reduce_min(G), 'G max:', tf.reduce_max(G))
        # 2. LUT에서 G와 가장 가까운 값을 찾음 (broadcasting 문제 해결)
        diff_LTP = np.abs(self.LUT_LTP[:, np.newaxis, np.newaxis] - G)  
        P_index_LTP = np.argmin(diff_LTP, axis=0).astype(int)  

        # 3. 인덱스 값이 범위를 초과하지 않도록 제한
        P_index_LTP = np.clip(P_index_LTP, 0, len(self.LUT_LTP) - 1)

        # Shape 유지 검증
        assert P_index_LTP.shape == G.shape, f"Shape mismatch: {P_index_LTP.shape} vs {G.shape}"

        # Debugging 출력
        tf.print('P_index_LTP min:', tf.reduce_min(P_index_LTP), 'P_index_LTP max:', tf.reduce_max(P_index_LTP))

        # Pulse는 discrete count이므로 float32가 아니라 int32로 변환해야 함
        return P_index_LTP



    @tf.function
    def pulse_update(self, gradient):
        alpha = 0.001  # Scaling factor for gradient step size

        if gradient is None:
            return
        

        tf.print("gradient min:", tf.reduce_min(gradient), "gradient max:", tf.reduce_max(gradient))

        # Gmax-Gmin 기반 정규화
        delta_P = alpha * (gradient / (self.Gmax - self.Gmin)) * (self.Pmax / 2)
        tf.print('delta_P min:', tf.reduce_min(delta_P), 'delta_P max:', tf.reduce_max(delta_P))
        #tf.print('delta_P',delta_P)
        # 기존 pulse 값 가져오기
        old_P = self.P

        # LTD 적용할 때 100으로 점프하는 조건 추가
        new_P = tf.where((old_P < 100) & (delta_P < 0), 100 + tf.abs(delta_P), old_P + delta_P)

        # Pulse 범위 클리핑 (0~Pmax)
        new_P = tf.clip_by_value(new_P, 0, self.Pmax)
        new_P = tf.round(new_P)

        # LTD 적용 시 100을 더한 경우, 다시 100을 빼서 원래 범위(0~100)로 조정
        new_P = tf.where(new_P >= 100, new_P - 100, new_P)

        new_P = tf.where(tf.math.is_nan(new_P), tf.ones_like(new_P) * self.Pmax / 4, new_P)
        tf.print('new_P min:', tf.reduce_min(new_P), 'new_P max:', tf.reduce_max(new_P))
        # LUT 기반 weight 조회
        new_weight = self.nonlinear_lookup(new_P)
        new_weight = tf.cast(new_weight, tf.float32)
        tf.print('new_weight min:', tf.reduce_min(new_weight), 'new_weight max:', tf.reduce_max(new_weight))
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
        convolutional_net = Sequential()
        """
        # Conv2D Layer 구성
        for i, kernel_size in zip(range(4), [(13, 13), (11, 11), (7, 7), (5, 5)]):
            convolutional_net.add(PulseBasedConv2D(
                filters=64 if i < 3 else 128,
                kernel_size=kernel_size,
                activation='relu',
                kernel_regularizer=l2(l2_regularization_penalization[f'Conv{i+1}']),
                name=f'Conv{i+1}'
            ))
            convolutional_net.add(MaxPool2D())
        """
        for i, kernel_size in zip(range(4), [(13, 13), (11, 11), (7, 7), (5, 5)]):
            convolutional_net.add(tf.keras.layers.Conv2D(
                filters=64 if i < 3 else 128,
                kernel_size=kernel_size,
                activation='relu',
                kernel_regularizer=l2(l2_regularization_penalization[f'Conv{i+1}']),
                name=f'Conv{i+1}'
            ))
            convolutional_net.add(MaxPool2D())
        
        convolutional_net.add(Flatten())

        # Dense Layer 구성
        convolutional_net.add(PulseBasedDense(
            units=1000, activation='tanh',
            kernel_regularizer=l2(l2_regularization_penalization['Dense1']),
            name='Dense1'
        ))
        convolutional_net.add(PulseBasedDense(
            units=100, activation='tanh',
            name='Dense3'
        ))

        # Siamese Network의 두 개의 입력
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        # 두 입력 이미지를 같은 네트워크에 통과시킴
        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L2 Distance Layer 생성
        l2_distance_layer = Lambda(lambda tensors: tf.expand_dims(
            K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1)), axis=-1))
        l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])

        # 모델 정의
        self.model_train = Model(inputs=[input_image_1, input_image_2], outputs=l2_distance)
        self.model_test = Model(inputs=input_image_1, outputs=encoded_image_1)


    def apply_sgd_update(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model_train(images, training=True)

            # 1. Predictions 디버깅
            tf.print("predictions min:", tf.reduce_min(predictions), "predictions max:", tf.reduce_max(predictions))
            if tf.math.is_nan(tf.reduce_sum(predictions)):
                raise ValueError("Predictions contain NaN! Check model output!")

            loss = self.contrastive_loss_w_margin(self.margin)(labels, predictions)
            loss = tf.reduce_mean(loss)

            # 2. Loss 디버깅
            tf.print("Loss value:", loss)
            if tf.math.is_nan(loss):
                raise ValueError("Loss is NaN! Check loss function and predictions!")
                
        # 3. Gradient 계산
        gradients = tape.gradient(loss, self.model_train.trainable_variables)

        # 4. Gradient NaN 확인
        gradient_nan_check = [tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.float32)) for g in gradients if g is not None]
        tf.print("Number of NaNs in gradients:", tf.reduce_sum(gradient_nan_check))
    
        if any(tf.math.is_nan(tf.reduce_sum(g)) for g in gradients if g is not None):
            raise ValueError("Gradient contains NaN!")

        # 5. Gradient Clipping (NaN 방지)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients if g is not None]

        self.optimizer.apply_gradients(zip(gradients, self.model_train.trainable_variables))

        return gradients, loss


    def train_step(self, gradients, optimizer):
        # Conv2D 레이어의 학습 가능한 변수 가져오기 (PulseBasedConv2D 포함)
        conv2d_vars = [var for layer in self.model_train.layers if isinstance(layer, Sequential)
                       for sub_layer in layer.layers if isinstance(sub_layer, (tf.keras.layers.Conv2D, PulseBasedConv2D))
                       for var in sub_layer.trainable_variables]

        conv2d_var_refs = [v.ref() for v in conv2d_vars]  # 변수 참조 리스트 생성

        # Conv2D 가중치는 SGD로 업데이트
        optimizer.apply_gradients(zip([grad for grad, var in zip(gradients, self.model_train.trainable_variables)
                                       if var.ref() in conv2d_var_refs], conv2d_vars))

        # Pulse-Based Conv2D & Dense Layer는 custom pulse update 적용
        for (grad, var) in zip(gradients, self.model_train.trainable_variables):
            for layer in self.model_train.layers:
                if isinstance(layer, Sequential):  
                    for sub_layer in layer.layers:
                        if isinstance(sub_layer,  PulseBasedDense) and sub_layer.kernel is var:
                            sub_layer.pulse_update(grad)
                elif isinstance(layer,  PulseBasedDense) and layer.kernel is var:
                    layer.pulse_update(grad)

    

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

        for iteration in range(number_of_iterations):
            images, labels = self.omniglot_loader.get_train_batch()

            # 먼저 SGD 기반 업데이트 적용 (gradient 반환)
            gradients,loss = self.apply_sgd_update(images, labels)
            #tf.print('grad',gradients)
            # Pulse-based weight update 적용
            self.train_step(gradients,self.optimizer)

            current_lr = K.get_value(self.learning_rate)
            print(f"Iteration {iteration}/{number_of_iterations} - Train Loss: {loss.numpy():.6f}, LR: {current_lr:.3E}")

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
                    self.model_test.save_weights(f'models/{model_name}.h5')

            if (iteration + 1) % evaluate_each == 0:
                evaluation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model_test, support_set_size, 40, is_validation=False)
                eval_accuracies[eval_count] = evaluation_accuracy
                eval_count += 1

            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' + str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Training Completed!')
        return best_validation_accuracy, eval_accuracies
