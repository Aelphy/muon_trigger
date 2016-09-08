"""generator that yields minibatches of data"""

import numpy as np

def constrained_sum_sample_pos(n, total):
    """
    Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur.
    """
    #dividers = sorted(np.random.choice(xrange(1, total), size=n - 1))
    #return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    dividers = np.array([total // n] * n)

    if total % n != 0:
        dividers[np.random.choice(np.arange(dividers.shape[0]), size=total % n)] += 1

    return dividers


def minibatch_generator(data, batch_size):
    """
        yields (Xideal, Xreal) minibatches
        data should be like [Xi, Xr]
            Xi, Xr are like {y: [I1, I2]}
    """

    def selector(batch_size, quantities):
        def f(x):
            result = []

            for i, key in enumerate(x.keys()):
                result.append(x[key][np.random.choice(np.arange(len(x[key])), size=quantities[i])])

            return np.vstack(result)

        return f

    while True:
        quantities = constrained_sum_sample_pos(len(data[0]), batch_size)
        yield list(map(selector(batch_size, quantities), data))


def iterate_minibatches(generator, n_minibatches):
    """like generator, but stops after n_minibatches"""
    for i in xrange(n_minibatches): yield generator.next()
