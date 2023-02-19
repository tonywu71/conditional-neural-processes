from functools import partial
from typing import Tuple, Callable, Iterator

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from dataloader.regression_data_generator_base import RegressionDataGeneratorBase


DEFAULT_TESTING_NUM_TARGET = 400


def gen_from_arbitrary_gp(
        batch_size,
        iterations,
        kernel_length_scale,
        min_num_context,
        max_num_context,
        min_num_target,
        max_num_target,
        min_x_val_uniform,
        max_x_val_uniform,
        testing) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Generates a batch of data for regression based on the original Conditional Neural Processes paper.
    Note that the data is generated batch-wise.
    
    During training and for each batch:
    - Both num_context and num_target are drawn from uniform distributions
    - The (num_context + num_target) x_values are drawn from a uniform distribution
    - A Gaussian Process with predefined kernel and a null mean function is used to generate the y_values from the x_values
    """    
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
    
    for _ in range(iterations):
        # NB: The distribution of y_values is the same for each iteration (i.e. the the one defined by
        #     the arbitrary GP) but the sampled x_values do differ (in terms of size and values).
        num_context = tf.random.uniform(shape=[],
                                    minval=min_num_context,
                                    maxval=max_num_context,
                                    dtype=tf.int32)

        if not testing:
            num_target = tf.random.uniform(shape=[],
                                        minval=min_num_target,
                                        maxval=max_num_target,
                                        dtype=tf.int32)
        else:
            # If testing, we want to use a fixed number of points for the target
            num_target = DEFAULT_TESTING_NUM_TARGET
        
        num_total_points = num_context + num_target
        
        x_values = tf.random.uniform(shape=(batch_size, num_total_points, 1),
                                     minval=min_x_val_uniform,
                                     maxval=max_x_val_uniform)
        
        gp = tfd.GaussianProcess(kernel, index_points=x_values, jitter=1.0e-4)
        y_values = tf.expand_dims(gp.sample(), axis=-1)
        
        idx = tf.random.shuffle(tf.range(num_total_points))
        
        # Select the targets which will consist of the context points
        # as well as some new target points
        target_x = x_values[:, :, :]
        target_y = y_values[:, :, :]

        # Select the observations
        context_x = tf.gather(x_values, indices=idx[:num_context], axis=1)
        context_y = tf.gather(y_values, indices=idx[:num_context], axis=1)

        yield (context_x, context_y, target_x), target_y


class RegressionDataGeneratorArbitraryGP(RegressionDataGeneratorBase):
    """Class that generates a batch of data for regression based on
    the original Conditional Neural Processes paper."""
    def __init__(self,
                 iterations: int=250,
                 batch_size: int=32,
                 min_num_context: int=3,
                 max_num_context: int=10,
                 min_num_target: int=2,
                 max_num_target: int=10,
                 min_x_val_uniform: int=-2,
                 max_x_val_uniform: int=2,
                 kernel_length_scale: float=0.4):
        super().__init__(iterations=iterations, batch_size=batch_size)
        
        assert min_num_context < max_num_context, "min_num_context must be smaller than max_num_context"
        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        
        assert min_num_target < max_num_target, "min_num_target must be smaller than max_num_target"
        self.min_num_target = min_num_target
        self.max_num_target = max_num_target
        
        assert min_x_val_uniform < max_x_val_uniform, "min_val_uniform must be smaller than max_val_uniform"
        self.min_x_val_uniform = min_x_val_uniform
        self.max_x_val_uniform = max_x_val_uniform
        
        self.kernel_length_scale = kernel_length_scale
        
        self.train_ds, self.test_ds = self.load_regression_data()

    
    def get_gp_curve_generator_from_uniform(self, testing: bool=False) -> Callable:
        """Returns a function that generates a batch of data for regression based on
        the original Conditional Neural Processes paper."""
        return partial(gen_from_arbitrary_gp,
                       batch_size=self.batch_size,
                       iterations=self.iterations,
                       kernel_length_scale=self.kernel_length_scale,
                       min_num_context=self.min_num_context,
                       max_num_context=self.max_num_context,
                       min_num_target=self.min_num_target,
                       max_num_target=self.max_num_target,
                       min_x_val_uniform=self.min_x_val_uniform,
                       max_x_val_uniform=self.max_x_val_uniform,
                       testing=testing)
    
    
    def load_regression_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Returns a tuple of training and test datasets."""
        train_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator_from_uniform(testing=False),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        test_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator_from_uniform(testing=True),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # No need to shuffle as the data is already generated randomly
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return train_ds, test_ds