from keras.layers import Dense, Flatten, merge, Merge, Lambda, RepeatVector, Subtract
from keras.models import Sequential, load_model, Model, Input, clone_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
import os
from abc import ABC, abstractmethod
from keras import backend as K
import numpy as np

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


class AbstractModel(ABC):

    def __init__(self, state_size=None, action_size=None, log_dir='.', weight_name='.', lr=0.001):
        self.model = None
        self.state_size = state_size
        self.action_size = action_size
        self.optim = Adam(lr=lr)

    @abstractmethod
    def _model_creation(self, **kwargs):
        pass

    def fit(self, x, y, batch=None, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batch, epochs=epochs, verbose=verbose)
        self.on_fit_end()

    def predict(self, x, batch=None, verbose=0):
        return self.model.predict(x, batch_size=batch, verbose=verbose)

    # def _load_model(self, model_name):
    #     return load_model(os.path.join(self.log_dir, model_name))
    #
    # def save_weight(self, name):
    #     self.model.save_weights(os.path.join(self.log_dir, name))
    #
    # def _load_weights(self, name):
    #     self.model.load_weights(os.path.join(self.log_dir, name))

    def on_fit_end(self):
        pass


class DoubleDQNWrapper(AbstractModel):
    def __init__(self, network, update_time):

        # self.inner_net = network
        super(DoubleDQNWrapper, self).__init__()

        self.model, self._target_net = self._model_creation(network)
        self.delta = 0.01
        self._time = 0
        self.update_time = update_time
        self.state_size = network.state_size
        self.action_size = network.action_size

    def on_fit_end(self):
        self._time += 1
        if self._time >= self.update_time:
            self._clone_weights()
            self._time = 0
        # mod_layer = self.model.layers
        # oth_layer = self._target_net.layers
        # for i in range(1, len(self.model.layers)):
        #     w = []
        #     for j in range(len(mod_layer[i].get_weights())):
        #         w1 = mod_layer[i].get_weights()[j]
        #         w2 = oth_layer[i].get_weights()[j]
        #         w.append(self.delta*w1 + (1-self.delta)*w2)
        #     oth_layer[i].set_weights(w)

    def predict(self, x, x_t=None, batch=None, verbose=0):

        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online
        target = self._target_net.predict(x_t, batch_size=batch, verbose=verbose)
        return online, target

    def _model_creation(self, network):
        model = network.model
        target_net = clone_model(network.model)
        target_net.set_weights(network.model.get_weights())
        return model, target_net

    def _clone_weights(self):
        self._target_net = clone_model(self.model)
        self._target_net.set_weights(self.model.get_weights())


class DenseDQN(AbstractModel):
    def __init__(self, log_dir, weight_name=None, action_size=None, state_size=None, lr=0.001, layer_size=(16, 16)):
        super(DenseDQN, self).__init__(state_size, action_size, log_dir, weight_name=weight_name, lr=lr)
        self.learning_rate = lr
        self.model = self._model_creation(layer_size)

    def _model_creation(self, layers_size=(), **kwargs):

        assert (len(layers_size) > 0)
        assert(self.state_size is not None and self.action_size is not None)

        model = Sequential()

        model.add(Dense(layers_size[0], input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        for i in layers_size[1:]:
            model.add(Dense(i, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self.optim)
        return model


class DuelDenseDQN(AbstractModel):
    def __init__(self, log_dir, weight_name=None, action_size=None, state_size=None, lr=0.001,
                 layer_size=(16, 16), layer_size_val=None):

        super(DuelDenseDQN, self).__init__(state_size, action_size, log_dir, weight_name=weight_name, lr=lr)

        self.learning_rate = lr
        self._time = 0
        self.model = self._model_creation(layer_size, layer_size_val)
        self._target_model = self.model

    def _model_creation(self, layers_size, layers_size_val):
        assert(len(layers_size) > 0)

        if layers_size_val is None:
            layers_size_val = layers_size
        assert(self.state_size is not None and self.action_size is not None)

        in_layer = Input(shape=(self.state_size, ))

        last_layer_adv = in_layer
        last_layer_val = in_layer

        for i in layers_size:
            last_layer_adv = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_adv)

        for i in layers_size_val:
            last_layer_val = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_val)

        advantage_layer = Dense(self.action_size, kernel_initializer='he_uniform',
                                activation='linear')(last_layer_adv)

        value_layer = Dense(1, kernel_initializer='he_uniform', activation='linear')(last_layer_val)

        value_layer = Lambda(function=lambda x: K.repeat_elements(x, 2, axis=1),
                             output_shape=lambda s: s)(value_layer)

        pol_out = Subtract()([advantage_layer, value_layer])
        model = Model(inputs=in_layer, outputs=pol_out)

        model.compile(loss='mse', optimizer=self.optim)
        return model
