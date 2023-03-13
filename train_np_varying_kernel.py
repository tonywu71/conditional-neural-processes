#%%
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('example-cnp/')

from dataloader.load_regression_data_from_arbitrary_gp_varying_kernel import RegressionDataGeneratorArbitraryGPWithVaryingKernel
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
#from model import ConditionalNeuralProcess as NeuralProcess
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

args = argparse.Namespace(epochs=200, batch=32, task='regression', num_context=10, uniform_sampling=True, model='LNP')
#args = argparse.Namespace(epochs=15, batch=128, task='celeb', num_context=10, uniform_sampling=False, model='HNP')

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs
TRAINING_ITERATIONS = 100
TEST_ITERATIONS = int(TRAINING_ITERATIONS/5)

model_path = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"


if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]


elif args.task == 'regression':
    data_generator = RegressionDataGeneratorArbitraryGPWithVaryingKernel(
        iterations=TRAINING_ITERATIONS,
        batch_size=BATCH_SIZE,
        min_num_context=3,
        max_num_context=40,
        min_num_target=2,
        max_num_target=40,
        min_x_val_uniform=-2,
        max_x_val_uniform=2,
        min_kernel_length_scale=0.1,
        max_kernel_length_scale=1.0
    )
    train_ds, test_ds = data_generator.load_regression_data()

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]

    

elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 6]









#%%

# Define NP Model
if args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)

optimizer = tf.keras.optimizers.Adam(1e-3)

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'np', args.model, args.task, time)
writer = tf.summary.create_file_writer(log_dir)
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True)
callbacks = [plot_clbk, cp_callback]
for callback in callbacks: callback.model = model


#%%


def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    


    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.math.reduce_mean(loss)


callbacks.append(
    tf.keras.callbacks.EarlyStopping(
        patience=25,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True)
    )


#%%

epochs = args.epochs
for epoch in range(0, epochs + 1):
    with tqdm(total=TRAINING_ITERATIONS, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        train_loss = tf.keras.metrics.Mean()
        for idx, train_x in enumerate(train_ds):

            # if idx == 5:
            #     tf.profiler.experimental.start(log_dir)
            #     print("profiling!")
            # tf.summary.trace_on(graph=True) # Uncomment to trace the computational graph

            loss = train_step(model, train_x, optimizer)

            # if idx == 5:
            #     tf.profiler.experimental.stop(log_dir)
            #     print("profiling end!")

            train_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Train Loss': train_loss.result().numpy()})
            tepoch.update(1)
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch*TRAINING_ITERATIONS + idx)
                # tf.summary.trace_export(
                #     name=f"NP_trace {idx}",
                #     step=idx) # Uncomment to trace the computational graph

            


        test_loss = tf.keras.metrics.Mean()
        for idx, test_x in enumerate(test_ds):
            loss = model.compute_loss(test_x)

            test_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Test Loss': test_loss.result().numpy()})
            with writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch*TEST_ITERATIONS + idx)

        tepoch.set_postfix({'Train loss': train_loss.result().numpy(), 'Test loss': test_loss.result().numpy()})
        
        for callback in callbacks: callback.on_epoch_end(epoch, logs={'loss': train_loss.result()})

#%%


