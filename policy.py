import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, policy_name):
        self.name = policy_name
        pass

    @abstractmethod
    def select_action(self, **kwargs):
        pass

    @abstractmethod
    def get_value(self):
        pass


class GreedyPolicy(Policy):
    def __init__(self):
        super(GreedyPolicy, self).__init__('greedy')

    def select_action(self, q_values):
        action = np.argmax(q_values)
        return action

    def get_value(self):
        return 'Greedy'


class RandomPolicy(Policy):
    def __init__(self):
        super(RandomPolicy, self).__init__('greedy')

    def select_action(self, q_values):
        action = np.random.choice(q_values)
        return action

    def get_value(self):
        return 'Random'


class EpsPolicy(Policy):
    def __init__(self, eps=.1, other_pol=None):
        super(EpsPolicy, self).__init__('epsGreedy')
        self.eps = eps
        if other_pol is None:
            other_pol = RandomPolicy
        self.other_pol = other_pol

    def select_action(self, q_values):

        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = self.other_pol.select_action(q_values)
        return action

    def get_value(self):
        return 'Eps: {}, Other pol: {}'.format(self.eps, self.other_pol.get_value())


class AnnealedPolicy(Policy):

    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps, train=True,
                 lower_prob_policy=None):

        assert (hasattr(inner_policy, attr))

        super(AnnealedPolicy, self).__init__('annealedPolicy')

        if lower_prob_policy is None:
            lower_prob_policy = GreedyPolicy()

        self.lower_prob_policy = lower_prob_policy
        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps
        self.train = train
        self.step = 0

    def get_current_value(self):
        if self.train:
            value = self._calcualte_value()
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):

        # v = self._calcualte_value()
        # if np.random.uniform() < v:
        setattr(self.inner_policy, self.attr, self._calcualte_value())
        self.step += 1
        return self.inner_policy.select_action(**kwargs)
        # else:
        #     return self.lower_prob_policy.select_action(**kwargs)

    def _calcualte_value(self):
        a = -float(self.value_max - self.value_min) / float(self.nb_steps)
        b = float(self.value_max)
        value = max(self.value_min, a * float(self.step) + b)
        return value

    def get_value(self):
        return 'Annealed  with inner: {}'.format(self.inner_policy.get_value())


class BoltzmannPolicy(Policy):
    def __init__(self, kt=1., limits=(-500., 500.)):
        super(BoltzmannPolicy, self).__init__('bolzemanPolicy')
        self.kt = kt
        self.limits = limits

    def select_action(self, q_values):

        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        exp_values = np.exp(np.clip(q_values / self.kt, self.limits[0], self.limits[1]))
        #Softmax
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_value(self):
        return '{} {}'.format(self.name, self.kt)


class BoltzmannExplorationPolicy(Policy):
    def __init__(self):
        super(BoltzmannExplorationPolicy, self).__init__('bolzemanExplorationPolicy')

        self.bolz = BoltzmannPolicy()
        self.exploration = AnnealedPolicy(inner_policy=EpsPolicy(1.0), attr='eps', value_max=1.0, value_min=0.1,
                                          value_test=0.5, nb_steps=1000)

    def select_action(self, q_values):
        if np.random.uniform() > self.exploration.get_value():
            return self.bolz.select_action(q_values=q_values)
        else:
            return self.exploration.select_action(q_values=q_values)

    def get_value(self):
        return self.name+' '+str(self.exploration.get_value())


if __name__ == '__main__':
    eps = EpsPolicy(1.0)

    ann = AnnealedPolicy(inner_policy=eps, attr='eps', value_max=1.0, value_min=0.1, value_test=0.5, nb_steps=1000)
    for i in range(1100):
        print(ann.get_current_value(i))
