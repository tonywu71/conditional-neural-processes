from functools import partial
from typing import Tuple, Callable, Iterator

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

tfd = tfp.distributions

def gen(batch_size, iterations, kernel_length_scale,
        min_num_context, max_num_context, min_num_target,
        testing) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:        
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
    
    for _ in range(iterations):
        num_context = tf.random.uniform(shape=[],
                                    minval=min_num_context,
                                    maxval=max_num_context,
                                    dtype=tf.int32)

        num_target = tf.random.uniform(shape=[],
                                    minval=min_num_target,
                                    maxval=max_num_context,
                                    dtype=tf.int32)
        
        num_total_points = num_context + num_target
        
        x_values = tf.random.uniform(shape=(batch_size, num_total_points, 1),
                                        minval=-2,
                                        maxval=2)
        
        gp = tfd.GaussianProcess(kernel, index_points=x_values, jitter=1.0e-4)
        y_values = tf.expand_dims(gp.sample(), axis=-1)
        
        if testing is True:
            num_target = 400
            num_total_points = num_target
            x_values = tf.expand_dims(tf.range(-2., 2., 1./100., dtype=tf.float32), axis=0)  # (1, 400)
            x_values = tf.tile(x_values, [batch_size, 1])  # (batch_size, 400)
            x_values = tf.expand_dims(x_values, axis=-1)  # (batch_size, 400, 1)
        else:
            x_values = x_values
            num_target = num_target
        
        y_values = tf.expand_dims(gp.sample(), axis=-1)

        if testing is True:
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random.shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)
        
        else:
            # Select the targets which will consist of the context points
            # as well as some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        yield (context_x, context_y, target_x), target_y


def get_gp_curve_generator_from_uniform(iterations: int=10000,
                                        batch_size: int=64,
                                        kernel_length_scale: float=0.4,
                                        min_num_context: int=3,
                                        max_num_context: int=10,
                                        min_num_target: int=2,
                                        testing: bool=False) -> Callable:
    
    return partial(gen,
                   batch_size=batch_size,
                   iterations=iterations,
                   kernel_length_scale=kernel_length_scale,
                   min_num_context=min_num_context,
                   max_num_context=max_num_context,
                   min_num_target=min_num_target,
                   testing=testing)


def load_regression_data(iterations: int=250, batch_size: int=32,
                         min_num_context: int=3, max_num_context: int=10, min_num_target: int=2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator_from_uniform(
            iterations=iterations,
            batch_size=batch_size,
            min_num_context=min_num_context,
            max_num_context=max_num_context,
            min_num_target=min_num_target,
            testing=False),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )
    test_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator_from_uniform(
            iterations=iterations,
            batch_size=batch_size,
            min_num_context=min_num_context,
            max_num_context=max_num_context,
            min_num_target=min_num_target,
            testing=True),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )
    
    train_ds = train_ds.shuffle(buffer_size=10).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_ds, test_ds



class RegressionDataGenerator():
    """Class that uses load_regression_data to create datasets.
    """
    def __init__(self, iterations: int=250, batch_size: int=32,
                 min_num_context: int=3, max_num_context: int=10, min_num_target: int=2):
        self.iterations = iterations
        self.batch_size = batch_size
        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        self.min_num_target = min_num_target
        
        self.train_ds, self.test_ds = load_regression_data(
            iterations=iterations, batch_size=batch_size,
            min_num_context=min_num_context,
            max_num_context=max_num_context,
            min_num_target=min_num_target)
    
    
    @staticmethod
    def plot_batch(context_x, context_y, target_x, target_y, figsize=(8, 5)):
        context_x = context_x.numpy()
        context_y = context_y.numpy()
        target_x = target_x.numpy()
        target_y = target_y.numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(target_x[0, :, 0], target_y[0, :, 0], c="blue", label='Target')
        ax.scatter(context_x[0, :, 0], context_y[0, :, 0], marker="x", c="red", label='Observations')
        ax.legend()

        return fig, ax
    
    
    def plot_random_batch(self, figsize=(8, 5)):
        """Plot a random batch from the train_ds.
        """
        (context_x, context_y, target_x), target_y = next(iter(self.train_ds.take(1)))
        fig, ax = RegressionDataGenerator.plot_batch(context_x, context_y, target_x, target_y, figsize=figsize)
        return fig, ax