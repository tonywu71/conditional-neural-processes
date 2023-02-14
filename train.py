#%%
import os
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_uniform import RegressionDataGeneratorUniform
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from model import ConditionalNeuralProcess
from utility import PlotCallback

tfk = tf.keras
tfd = tfp.distributions

# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
# parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size for training')
# parser.add_argument('-t', '--task', type=str, default='mnist', help='Task to perform : (mnist|regression)')

# args = parser.parse_args()

#tf.config.set_visible_devices([], 'GPU')

args = argparse.Namespace(epochs=15, batch=64, task='mnist', num_context=1, uniform_sampling=False)

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"


if args.task == 'mnist':
    train_ds, test_ds = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

elif args.task == 'regression':
    data_generator = RegressionDataGeneratorUniform()
    train_ds, test_ds = data_generator.load_regression_data(batch_size=BATCH_SIZE)

    # Model architecture
    encoder_dims = [128, 128, 128, 128]
    decoder_dims = [128, 128, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

elif args.task == 'celeb':
    train_ds, test_ds = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

# Compile model
model = ConditionalNeuralProcess(encoder_dims, decoder_dims)
model.compile(loss=loss, optimizer='adam')

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'cnp', args.task, time)
tensorboard_clbk = tfk.callbacks.TensorBoard(
    log_dir=log_dir, update_freq='batch')
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)
callbacks = [tensorboard_clbk, plot_clbk, cp_callback]


# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)







#%%


import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 4, figsize=(10, 5))
for i, num_context in enumerate([1,10,100,1000]):#([1,10,100,1000]):

    model.load_weights(f'.data/model_{args.task}_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")

    

    if args.task == 'celeb':
        train_ds, test_ds = load_celeb(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
        img_size=32

        it = iter(test_ds)
        next(it)
        # next(it)
        # next(it)
        (context_x, context_y, target_x), target_y = next(it)
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        # Plot context points
        white_img = tf.tile(tf.constant([[[0.,0.,0.]]]), [img_size, img_size, 1])
        indices = tf.cast(context_x[0] * float(img_size - 1.0), tf.int32)

        updates = context_y[0]

        context_img = tf.tensor_scatter_nd_update(white_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title(f'{num_context} context points')
        # Plot mean and variance
        mean = tf.reshape(mu[0], (img_size, img_size, 3))
        var = tf.reshape(sigma[0], (img_size, img_size, 3))

        axs[1][i].imshow(mean.numpy(), vmin=0., vmax=1.)
        axs[2][i].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[1][i].axis('off')
        axs[2][i].axis('off')
        axs[1][i].set_title('Predicted mean')
        axs[2][i].set_title('Predicted variance')

    elif args.task == 'mnist':
        train_ds, test_ds = load_mnist(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
        img_size=28
        
        (context_x, context_y, target_x), target_y = next(iter(test_ds))
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        # Plot context points
        blue_img = tf.tile(tf.constant([[[0.,0.,1.]]]), [28, 28, 1])
        indices = tf.cast(context_x[0] * 27., tf.int32)
        updates = tf.tile(context_y[0], [1, 3])
        context_img = tf.tensor_scatter_nd_update(blue_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title('Given context')
        # Plot mean and variance
        mean = tf.tile(tf.reshape(mu[0], (28, 28, 1)), [1, 1, 3])
        var = tf.tile(tf.reshape(sigma[0], (28, 28, 1)), [1, 1, 3])
        axs[1][i].imshow(mean.numpy(), vmin=0., vmax=1.)
        axs[2][i].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[1][i].axis('off')
        axs[2][i].axis('off')
        axs[1][i].set_title('Predicted mean')
        axs[2][i].set_title('Predicted variance')
        

# %%
num_context = 10

model.load_weights(f'.data/model_{args.task}_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0008.ckpt")

    
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3,2, figsize=(10, 5))
if args.task == 'celeb':
    for i, uniform in enumerate([True, False]):
        train_ds, test_ds = load_celeb(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=uniform)
        img_size=32

        it = iter(test_ds)
        next(it)
        # next(it)
        # next(it)
        (context_x, context_y, target_x), target_y = next(it)
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        # Plot context points
        white_img = tf.tile(tf.constant([[[0.,0.,0.]]]), [img_size, img_size, 1])
        indices = tf.cast(context_x[0] * float(img_size - 1.0), tf.int32)

        updates = context_y[0]

        context_img = tf.tensor_scatter_nd_update(white_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title(f'{num_context} context points ' + ("uniform" if uniform else "sorted"))
        # Plot mean and variance
        mean = tf.reshape(mu[0], (img_size, img_size, 3))
        var = tf.reshape(sigma[0], (img_size, img_size, 3))

        axs[1][i].imshow(mean.numpy(), vmin=0., vmax=1.)
        axs[2][i].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[1][i].axis('off')
        axs[2][i].axis('off')
        axs[1][i].set_title('Predicted mean')
        axs[2][i].set_title('Predicted variance')