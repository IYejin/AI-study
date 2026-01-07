
# project_1 : CIFAR-10 CNN baseline (TensorFlow/Keras) + FLOPs
# project_1_cnn.py

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def compute_flops_keras_model(model: keras.Model, input_shape=(1, 32, 32, 3)) -> int:
    """
    Returns total float ops (FLOPs) for a single forward pass with given input_shape.
    Note: FLOPs definition depends on profiler; this uses TF v1 profiler float_operation.
    """
    # Make sure the model is built
    _ = model(tf.zeros(input_shape, dtype=tf.float32), training=False)

    # Build a concrete function
    @tf.function
    def forward(x):
        return model(x, training=False)

    concrete = forward.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

    try:
        # Works in most TF2 versions
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    except Exception as e:
        # Fallback: try convert_variables_to_constants_v2 then grab graph_def
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(concrete)
        graph_def = frozen_func.graph.as_graph_def()

    # Import into a fresh graph (TF1-style) and profile
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        total_flops = 0 if flops is None else flops.total_float_ops
    return int(total_flops)

def build_model():
    # Data Augmentation as layers (applied only during training automatically)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.35)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_cnn_baseline")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="project_1_cifar10_cnn.keras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="Optional: use only first N train samples for a quick run (0=use all).")
    args = parser.parse_args()

    # Reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print("TF version:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices('GPU'))

    # 1) Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Optional subset for quick tests
    if args.subset and args.subset > 0:
        x_train = x_train[:args.subset]
        y_train = y_train[:args.subset]

    # 2) Normalize (0~1)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # 4) Build model
    model = build_model()
    model.summary()

    # FLOPs / Params (inference)
    flops_b1 = compute_flops_keras_model(model, input_shape=(1, 32, 32, 3))
    params = model.count_params()
    print(f"Params: {params:,} ({params/1e6:.3f} M)")
    print(f"FLOPs (batch=1, forward): {flops_b1:,} ({flops_b1/1e9:.3f} GFLOPs)")
    FLOPs_batch128 = flops_b1 * 128

    # 5) Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 6) Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # 7) Train
    history = model.fit(
        x_train, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 8) Test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # 9) Save
    model.save(args.save_path)
    print("Saved:", args.save_path)

if __name__ == "__main__":
    main()

