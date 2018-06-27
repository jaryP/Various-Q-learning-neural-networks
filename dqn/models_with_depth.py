from keras.layers import Dense, Flatten, merge, Merge, Lambda, Add, Multiply, Flatten, Subtract
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.models import  load_model, Model, Input, clone_model
from keras.optimizers import Adam
import os
from abc import ABC, abstractmethod
from keras import backend as K
import numpy as np

def huber_loss(y_true, y_pred, clip_value=1.0):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)
        else:
            return tf.where(condition, squared_loss, linear_loss)
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)

class AbstractModel(ABC):

    def __init__(self, state_size=None, action_size=None, lr=0.001, depth=1, normalization=False):

        self.model = None
        self._target_net = None
        self.state_size = state_size
        self.action_size = action_size
        self.optim = Adam(lr=lr)
        self.depth = depth
        self.normalization=normalization

    @abstractmethod
    def _model_creation(self, **kwargs):
        pass

    def fit(self, x, y, batch=None, epochs=1, verbose=0):
        h = self.model.fit(x, y, batch_size=batch, epochs=epochs, verbose=verbose)
        self.on_fit_end()
        return h

    def predict(self, x, x_t=None, batch=None, verbose=0):
        online = self.model.predict(x, batch_size=batch, verbose=verbose)
        if x_t is None:
            return online
        if self._target_net is None:
            target = self.model.predict(x, batch_size=batch, verbose=verbose)
        else:
            target = self._target_net.predict(x_t, batch_size=batch, verbose=verbose)
        return online, target

    def save(self, path):
        self.model.save(os.path.join(path, 'model.h5'))
        self.model.save_weights(os.path.join(path, 'model_weights.h5'))

    def load(self, path):
        self. model = load_model(os.path.join(path, 'model.h5'), custom_objects={'huber_loss': huber_loss})
        self.model.load_weights(os.path.join(path, 'model_weights.h5'))
        # self.model.compile(self.optim, huber_loss)

    def on_fit_end(self):
        pass


class DoubleDQNWrapper(AbstractModel):
    def __init__(self, network, update_time):

        super(DoubleDQNWrapper, self).__init__()

        self.model, self._target_net = self._model_creation(network)
        # self._target_net = self.model
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

    def _model_creation(self, network):
        model = network.model
        target_net = clone_model(network.model)
        target_net.set_weights(network.model.get_weights())
        return model, target_net

    def _clone_weights(self):
        self._target_net = clone_model(self.model)
        self._target_net.set_weights(self.model.get_weights())

    def save(self, path):
        # self.model.save(os.path.join(path, 'model.h5'))
        # self.model.save_weights(os.path.join(path, 'model_weights.h5'))
        super().save(path)
        self._target_net.save(os.path.join(path, 'target_model.h5'))
        self._target_net.save_weights(os.path.join(path, 'target_model_weights.h5'))

    def load(self, path):
        # self.model = load_model(os.path.join(path, 'model.h5'),  custom_objects={'hube_loss': huber_loss})
        super().load(path)
        # self._target_net = load_model(os.path.join(path, 'target_model.h5'),  custom_objects={'huber_loss': huber_loss})
        self._target_net.load_weights(os.path.join(path, 'target_model_weights.h5'))


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

        in_layer = Input((self.depth, self.state_size, ))
        reg = in_layer

        if self.normalization:
            reg = Lambda(lambda x: x / 255.0, output_shape=(self.depth, self.state_size, ))(in_layer)

        mask = Input((self.action_size, ))

        last_layer = Flatten(data_format='channels_first')(reg)

        last_layer = Dense(layers_size[0], activation='relu', kernel_initializer='glorot_normal')(last_layer)
        for i in layers_size[1:]:
            last_layer = Dense(i, activation='relu', kernel_initializer='glorot_normal')(last_layer)

        # last_layer = Flatten(name='last_layer')(last_layer)
        last_layer = Dense(self.action_size)(last_layer)

        # last_layer = Lambda(lambda x: x[:, 0, :], output_shape=(self.action_size, ))(last_layer)

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

        in_layer = Input(shape=(self.depth, self.state_size, ))
        mask = Input((self.action_size, ))

        last_layer_adv = Flatten()(in_layer)
        last_layer_val = Flatten()(in_layer)

        for i in layers_size:
            last_layer_adv = Dense(i, activation='relu', kernel_initializer='glorot_normal')(last_layer_adv)

        for i in layers_size_val:
            last_layer_val = Dense(i, activation='relu', kernel_initializer='glorot_normal')(last_layer_val)

        advantage_layer = Dense(self.action_size, activation='linear',
                                kernel_initializer='glorot_normal')(last_layer_adv)

        # advantage_layer = Lambda(lambda x: x[:, 0, :], output_shape=(self.action_size, ))(Flatten()(advantage_layer))

        value_layer = Dense(1, activation='linear', kernel_initializer='glorot_normal')(last_layer_val)

        # value_layer = Lambda(lambda x: x[:, 0, :], output_shape=(self.action_size, ))(value_layer)

        value_layer = Lambda(function=lambda x: K.repeat_elements(x, self.action_size, axis=1),
                             output_shape=lambda s: s)(value_layer)

        pol_out = Add()([advantage_layer, value_layer])

        filtered_output = Multiply()([pol_out, mask])

        model = Model(inputs=[in_layer, mask], outputs=filtered_output)

        model.compile(loss='mse', optimizer=self.optim)
        return model


class ConvDQM(AbstractModel):
    def __init__(self, action_size=None, state_size=None, lr=0.001, depth=1):

        super(ConvDQM, self).__init__(state_size, action_size, depth=depth, lr=lr)

        self.learning_rate = lr
        self.model = self._model_creation()
        self.model._make_train_function()
        self.model._make_predict_function()

        # self._target_net = self.model

    def _model_creation(self, **kwargs):
        in_shape = (self.depth, self.state_size[0], self.state_size[1], )

        frames_input = Input(shape=in_shape, name='frames')
        mask = Input(shape=(self.action_size,), name='mask')

        normz = Lambda(lambda x: x / 255.0, output_shape=in_shape)(frames_input)

        conv = Conv2D(kernel_size=8, filters=16, strides=4, activation='relu',
                      data_format='channels_first')(normz)
        conv = Conv2D(kernel_size=4, filters=32, strides=2, activation='relu',
                      data_format='channels_first')(conv)
        # conv = Conv2D(filters=64, kernel_size=2, strides=2, activation='relu')(conv)
        flt = Flatten(data_format='channels_first')(conv)

        flt = Dense(256, activation='relu', name='last_layer')(flt)
        flt = Dense(self.action_size)(flt)

        flt = Multiply()([flt, mask])

        model = Model(input=[frames_input, mask], output=flt)
        # model = Model(input=frames_input, outputs=flt)
        model.compile(self.optim, loss=huber_loss)
        return model


class ConvDDQN(AbstractModel):
    def __init__(self, action_size=None, state_size=None, lr=0.001, depth=1):

        super(ConvDDQN, self).__init__(state_size, action_size, depth=depth, lr=lr)

        self.learning_rate = lr
        self._time = 0
        self.model = self._model_creation()
        # self._target_net = None

    def _model_creation(self, **kwargs):
        in_shape = (self.depth, self.state_size[0], self.state_size[1], )

        frames_input = Input(shape=in_shape, name='frames')
        mask = Input(shape=(self.action_size,), name='mask')

        normz = Lambda(lambda x: x / 255.0, output_shape=in_shape)(frames_input)

        conv = Convolution2D(kernel_size=(8, 8), filters=32, strides=(4, 4), activation='relu',
                             data_format='channels_first')(normz)
        conv = Convolution2D(kernel_size=(4, 4), filters=64, strides=(2, 2), activation='relu',
                             data_format='channels_first')(conv)
        conv = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(conv)
        flt = Flatten(data_format='channels_first')(conv)

        advantage_flt = Dense(256, activation='relu', )(flt)
        advantage = Dense(self.action_size, name='advantege_layer')(advantage_flt)

        advantage_sub = Lambda(function=lambda x: K.mean(x,  axis=1, keepdims=True))(advantage)

        value_flt = Dense(128, activation='relu', )(flt)
        value = Dense(1, name='value_layer')(value_flt)

        value = Lambda(function=lambda x: K.repeat_elements(x, self.action_size, axis=1))(value)

        out = Subtract()([advantage, advantage_sub])
        out = Add()([out, value])

        flt = Multiply()([out, mask])

        model = Model(input=[frames_input, mask], output=flt)
        # model = Model(input=frames_input, outputs=flt)
        model.compile(self.optim, loss=huber_loss)
        return model
