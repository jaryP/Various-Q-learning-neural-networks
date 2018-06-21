from keras.layers import Dense, Flatten, merge, Merge, Lambda, RepeatVector, Add, Multiply
from keras.models import Sequential, load_model, Model, Input, clone_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
import os
from abc import ABC, abstractmethod
from keras import backend as K
import numpy as np
from tensorflow import where


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = 1.0 * (K.abs(err) - 0.5 * 1.0)

    loss = where(cond, L2, L1)
    return K.mean(loss)


# def huber_loss(a, b):
#     error = a - b
#     quadratic_term = error*error / 2
#     linear_term = abs(error) - 1/2
#     use_linear_term = (abs(error) > 1.0)
#     use_linear_term = K.cast(use_linear_term, 'float32')
#
#     return K.mean(use_linear_term * linear_term + (1-use_linear_term) * quadratic_term)


class AbstractModel(ABC):

    def __init__(self, state_size=None, action_size=None, lr=0.001, depth=1):
        self.model = None
        self._target_net = self.model
        self.state_size = state_size
        self.action_size = action_size
        self.optim = RMSprop(lr=lr)
        self.depth = depth

    @abstractmethod
    def _model_creation(self, **kwargs):
        pass

    def fit(self, x, y, batch=None, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batch, epochs=epochs, verbose=verbose)
        self.on_fit_end()

    def predict(self, x, x_t=None, batch=None, verbose=0):
        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online
        target = self._target_net.predict(x_t, batch_size=batch, verbose=verbose)
        return online, target

    def save(self, path):
        self.model.save(os.path.join(path, 'model.h5'))
        self.model.save_weights(os.path.join(path, 'model_weights.h5'))

    def load(self, path):
        self. model = load_model(os.path.join(path, 'model.h5'))

    def on_fit_end(self):
        pass


class DoubleDQNWrapper(AbstractModel):
    def __init__(self, network, update_time):

        # self.inner_net = network
        super(DoubleDQNWrapper, self).__init__()

        self.model, self._target_net = self._model_creation(network)
        self._target_net = self.model
        self.delta = 0.01
        self._time = 0
        self.update_time = update_time
        self.state_size = network.state_size
        self.action_size = network.action_size
        self.depth = network.depth

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
    # def predict(self, x, x_t=None, batch=None, verbose=0):
    #     online = self.model.predict(x, batch_size=batch, verbose=verbose)
    #     if x_t is None:
    #         return online
    #
    #     target = self._target_net.predict(x_t, batch_size=batch, verbose=verbose)
    #     return online, target

    def _model_creation(self, network):
        model = network.model
        target_net = clone_model(network.model)
        target_net.set_weights(network.model.get_weights())
        return model, target_net

    def _clone_weights(self):
        self._target_net = clone_model(self.model)
        self._target_net.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save(os.path.join(path, 'model.h5'))
        self.model.save_weights(os.path.join(path, 'model_weights.h5'))
        self._target_net.save(os.path.join(path, 'model.h5'))
        self._target_net.save_weights(os.path.join(path, 'model_weights.h5'))

    def load(self, path):
        self.model = load_model(os.path.join(path, 'model.h5'))
        self._target_net= load_model(os.path.join(path, 'model.h5'))


class DenseDQN(AbstractModel):
    def __init__(self, action_size=None, state_size=None, lr=0.001, layer_size=(16, 16),
                 depth=1):
        super(DenseDQN, self).__init__(state_size, action_size, lr=lr, depth=depth)

        self.learning_rate = lr
        self.model = self._model_creation(layer_size)
        self._target_net = self.model

    def _model_creation(self, layers_size=(), **kwargs):

        assert (len(layers_size) > 0)
        assert(self.state_size is not None and self.action_size is not None)

        in_layer = Input((self.state_size, ))
        mask = Input((self.action_size, ))

        last_layer = in_layer

        last_layer = Dense(layers_size[0], activation='relu', kernel_initializer='he_uniform')(last_layer)
        for i in layers_size[1:]:
            last_layer = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer)
        last_layer = Dense(self.action_size, activation='linear')(last_layer)

        filtered_output = Multiply()([last_layer, mask])

        model = Model(inputs=[in_layer, mask], outputs=filtered_output)
        model.compile(loss='mse', optimizer=self.optim)
        return model


class DuelDenseDQN(AbstractModel):
    def __init__(self, action_size=None, state_size=None, lr=0.001, layer_size=(16, 16), layer_size_val=None, depth=1):

        super(DuelDenseDQN, self).__init__(state_size, action_size, depth=depth, lr=lr)

        self.learning_rate = lr
        self._time = 0
        self.model = self._model_creation(layer_size, layer_size_val)
        self._target_net = self.model

    def _model_creation(self, layers_size, layers_size_val):
        assert(len(layers_size) > 0)

        if layers_size_val is None:
            layers_size_val = layers_size
        assert(self.state_size is not None and self.action_size is not None)

        in_layer = Input(shape=(self.state_size, ))
        mask = Input((self.action_size, ))

        last_layer_adv = in_layer
        last_layer_val = in_layer

        for i in layers_size:
            last_layer_adv = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_adv)

        for i in layers_size_val:
            last_layer_val = Dense(i, activation='relu', kernel_initializer='he_uniform')(last_layer_val)

        advantage_layer = Dense(self.action_size, kernel_initializer='he_uniform',
                                activation='linear')(last_layer_adv)

        value_layer = Dense(1, kernel_initializer='he_uniform', activation='linear')(last_layer_val)

        value_layer = Lambda(function=lambda x: K.repeat_elements(x, self.action_size, axis=1),
                             output_shape=lambda s: s)(value_layer)

        pol_out = Add()([advantage_layer, value_layer])

        filtered_output = Multiply()([pol_out, mask])

        model = Model(inputs=[in_layer, mask], outputs=filtered_output)

        model.compile(loss='mse', optimizer=self.optim)
        return model
