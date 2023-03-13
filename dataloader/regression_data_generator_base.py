from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import matplotlib.pyplot as plt


class RegressionDataGeneratorBase(ABC):
    """Abstract base class for regression data generators."""
    def __init__(self,
                 iterations: int,
                 batch_size: int,
                 min_num_context: int,
                 max_num_context: int,
                 min_num_target: int,
                 max_num_target: int,
                 min_x_val_uniform: int,
                 max_x_val_uniform: int,
                 n_iterations_test: Optional[int]=None):
        self.iterations = iterations
        self.batch_size = batch_size
        
        assert min_num_context < max_num_context, "min_num_context must be smaller than max_num_context"
        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        
        assert min_num_target < max_num_target, "min_num_target must be smaller than max_num_target"
        self.min_num_target = min_num_target
        self.max_num_target = max_num_target
        
        assert min_x_val_uniform < max_x_val_uniform, "min_val_uniform must be smaller than max_val_uniform"
        self.min_x_val_uniform = min_x_val_uniform
        self.max_x_val_uniform = max_x_val_uniform
        
        if n_iterations_test is None:
            self.n_iterations_test = self.iterations // 10
        else:
            self.n_iterations_test = n_iterations_test
        
        # The following attributes will be set when calling load_regression_data() from
        # the child class:
        self.train_ds: tf.data.Dataset = None
        self.test_ds: tf.data.Dataset = None
    
    
    @abstractmethod
    def get_gp_curve_generator(self, testing: bool=False) -> Callable:
        """Returns a generator function that generates regression data from a Gaussian Process."""
        pass
    
    
    def load_regression_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Returns a tuple of training and test datasets."""
        train_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator(testing=False),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        test_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator(testing=True),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # No need to shuffle as the data is already generated randomly
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return train_ds, test_ds
    
    
    @staticmethod
    def plot_first_elt_of_batch(context_x, context_y, target_x, target_y,
                                ax: Optional[plt.Axes]=None,
                                figsize=(8, 5)):
        """Plot the first element of a batch."""
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        context_x = context_x.numpy()
        context_y = context_y.numpy()
        target_x = target_x.numpy()
        target_y = target_y.numpy()
        
        ax.scatter(target_x[0, :, 0], target_y[0, :, 0], c="blue", label='Target')
        ax.scatter(context_x[0, :, 0], context_y[0, :, 0], marker="x", c="red", label='Observations')
        ax.legend()

        return ax
    
    
    def plot_first_elt_of_random_batch(self, figsize=(8, 5)):
        """Plot a random batch from the training set."""
        (context_x, context_y, target_x), target_y = next(iter(self.train_ds.take(1)))
        ax = RegressionDataGeneratorBase.plot_first_elt_of_batch(context_x, context_y, target_x, target_y, figsize=figsize)
        return ax
