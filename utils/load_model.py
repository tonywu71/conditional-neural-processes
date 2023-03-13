from pathlib import Path
import tensorflow as tf
from neural_process_model_latent import NeuralProcessLatent


def load_lnp_model(model_path: str) -> tf.keras.Model:
    assert Path(model_path).parent.is_dir()

    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]

    model = NeuralProcessLatent(z_output_sizes=z_output_sizes,
                                enc_output_sizes=enc_output_sizes,
                                dec_output_sizes=dec_output_sizes)

    model.load_weights(model_path)

    return model
