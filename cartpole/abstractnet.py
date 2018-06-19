from keras.layers import Dense, Flatten, merge, Merge, Lambda
from keras.models import Sequential, load_model, Model, Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
import os
from abc import ABC, abstractmethod
from keras import backend as K
import numpy as np


def hubber_loss(a, b):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


class AbstractNet(ABC):
    def __init__(self, log_dir,  state_size, action_size, weight_name, lr):
        self.log_dir = log_dir
        self.learning_rate = lr

        #self.model = self._model_creation(state_size, action_size)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if weight_name is not None:
            self._load_weights(weight_name)


    @abstractmethod
    def _model_creation(self, state_size, action_size, ):
        self.model = None
        pass

    def fit(self, x, y, batch=None, epochs=1, callb=None, verbose=0):

        if callb is None:
            callb = list()

        '''initial_ep = 0
        if self.model_name is not None:
            initial_ep = int(self.model_name.rsplit('.')[0])'''

        #callb.extend([TensorBoard(log_dir=self.log_dir, batch_size=batch)])
        callb.extend([ModelCheckpoint(filepath=os.path.join(self.log_dir, '{epoch:04d}.hdf5'),
                                      monitor='loss')])
        callb=[]
        self.model.fit(x, y, batch_size=batch, epochs=epochs, callbacks=callb, verbose=verbose)

    def predict(self, x, batch=None, verbose=0):
        return self.model.predict(x, batch_size=batch, verbose=verbose)

    def _load_model(self, model_name):
        return load_model(os.path.join(self.log_dir, model_name))

    def save_weight(self, name):
        self.model.save_weights(os.path.join(self.log_dir, name))

    def _load_weights(self, name):
        self.model.load_weights(os.path.join(self.log_dir, name))


class DenseDQN(AbstractNet):
    def __init__(self, log_dir,  weight_name=None, action_size=None, state_size=None, lr=0.001, layer_size=(16, 16)):
        super(DenseDQN, self).__init__(log_dir, state_size=state_size,
                                       action_size=action_size, weight_name=weight_name, lr=lr)

        self.model = self._model_creation(state_size, action_size, layer_size)

    def _model_creation(self, state_size, action_size, layers_size=()):
        assert(len(layers_size) > 0)
        model = Sequential()

        model.add(Dense(layers_size[0], input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
        for i in layers_size[1:]:
            model.add(Dense(i, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


class DenseDDQN(AbstractNet):
    def __init__(self, log_dir, update_time=1, weight_name=None, action_size=None, state_size=None, lr=0.001,
                 layers_size=(16, 16)):
        super(DenseDDQN, self).__init__(log_dir, state_size=state_size,
                                        action_size=action_size, weight_name=weight_name, lr=lr)
        self._time = 0
        self.model = self._model_creation(state_size, action_size, layers_size)
        self._update_time = update_time
        self._target_model = self.model

    def _model_creation(self, state_size, action_size, layers_size=(16, 16)):
        assert(len(layers_size) > 0)
        model = Sequential()

        model.add(Dense(layers_size[0], input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
        for i in layers_size[1:]:
            model.add(Dense(i, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01))
        return model

    def predict(self, x, x_t=None, batch=None, verbose=0):

        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online

        target = self._target_model.predict(x_t, batch_size=batch, verbose=verbose)
        self._time += 1
        if 0 < self._update_time <= self._time:
            self.copy_weights()
            self._time = 0
        return online, target

    def copy_weights(self):
        self._target_model.set_weights(self.model.get_weights())


class DuelDenseDQN(AbstractNet):
    def __init__(self, log_dir, update_time=1, weight_name=None, action_size=None, state_size=None, lr=0.001,
                 layers_size_adv=(16, 16), layer_size_val=None):
        super(DuelDenseDQN, self).__init__(log_dir, state_size=state_size,
                                           action_size=action_size, weight_name=weight_name, lr=lr)

        self._time = 0
        self.model = self._model_creation(state_size, action_size, layers_size_adv, layer_size_val)
        self._update_time = update_time
        self._target_model = self.model

    def _model_creation(self, state_size, action_size, layers_size_adv, layers_size_val):
        print(layers_size_adv, layers_size_val)
        assert(len(layers_size_adv) > 0)

        if layers_size_val is None:
            layers_size_val = layers_size_adv

        in_layer = Input(shape=(state_size, ))

        last_layer_adv = in_layer
        last_layer_val = in_layer

        for i in layers_size_adv:
            last_layer_adv = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_adv)

        for i in layers_size_val:
            last_layer_val = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_val)

        advantage_layer = Dense(action_size, kernel_initializer='he_uniform',
                                activation='linear')(last_layer_adv)

        value_layer = Dense(1, kernel_initializer='he_uniform', activation='linear')(last_layer_val)

        # pol_out = merge([advantage_layer, value_layer],
        #                 mode=lambda x: x[0] - K.mean(x[0], axis=0) + K.repeat_elements(x[1], 2, axis=1),
        #                 output_shape=lambda s: (s[0][0], s[0][1]))
        pol_out = merge([advantage_layer, value_layer],
                        mode=lambda x: x[0] + K.repeat_elements(x[1], 2, axis=1),
                        output_shape=lambda s: (s[0][0], s[0][1]))

        model = Model(inputs=in_layer, outputs=pol_out)

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def predict(self, x, x_t=None, batch=None, verbose=0):

        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online

        target = self._target_model.predict(x_t, batch_size=batch, verbose=verbose)
        #target = target[0]
        self._time += 1
        if 0 < self._update_time <= self._time:
            self.copy_weights()
            self._time = 0
        return online, target

    def copy_weights(self):
        mod_layer = self.model.layers
        oth_layer = self._target_model.layers
        for i in range(1, len(self.model.layers)):
            w = []
            for j in range(len(mod_layer[i].get_weights())):
                w1 = mod_layer[i].get_weights()[j]
                w2 = oth_layer[i].get_weights()[j]
                w.append(0.001*w1 + (1-0.001)*w2)
            oth_layer[i].set_weights(w)


if __name__ == '__main__':
    a = np.random.random((32, 4))
    b = np.random.randint(2, size=(32, 2))
    print(b.shape)
    from theano.printing import Print

    in_layer = Input(shape=(4,))

    value_layer = Dense(2, kernel_initializer='he_uniform')((Dense(12))(in_layer))
    adv_layer = Dense(1, kernel_initializer='he_uniform')((Dense(12))(in_layer))

    pol = merge([value_layer, adv_layer], mode=lambda x: x[0]-K.mean(x[0], axis=0) + K.repeat_elements(x[1], 2, axis=1),
                output_shape= lambda s: (s[0][0], s[0][1]))

    def antirectifier(x):
        return x[0]-K.mean(x[0], axis=0)+x[1]

    def antirectifier_output_shape(input_shape):
        print(input_shape)
        s1 = input_shape[0]
        s2 = input_shape[1]
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] = 2
        print(s1, s2)
        return s1

    last = Lambda(antirectifier, output_shape=antirectifier_output_shape)([value_layer, adv_layer])

    model = Model(inputs=[in_layer], outputs=[pol])

    model.compile(loss='mse',
                  optimizer=Adam(lr=0.001))
    #a = DuelDenseDQN(log_dir='.', action_size=2, state_size=4, layers_size=(24, 16, 8))
    model.summary()
    model.fit(a, b, epochs=1)
