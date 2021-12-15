import tensorflow as tf
import os
import logging
import keras
from src.utils.all_utils import get_timestamp



def resnet(input_shape,model_path):
    model= tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,weights="imagenet",include_top=False)
    model.save(model_path)
    logging.info(f"resnet50 base model saved at:{model_path}")
    return model

def prepare_model(model,classes,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable= False
    
    elif (freeze_till is not None) and (freeze_till>0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable= False

    
    flatten_in= tf.keras.layers.Flatten()(model.output)
    fc1=tf.keras.layers.Dense(256,activation="relu")(flatten_in)
    fc2=tf.keras.layers.Dense(128,activation="relu")(fc1)
    prediction= tf.keras.layers.Dense(
        units=classes,
        activation="softmax"        
    )(fc2)


    full_model=tf.keras.models.Model(
        inputs= model.input,
        outputs= prediction
    )


    full_model.compile(

        optimizer= tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss= tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    logging.info("custom model is compiled and ready to be trained")
    full_model.summary()
    return full_model



def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from: {untrained_full_model_path}")
    return model
        
def get_unique_path_to_save_model(trained_model_dir, model_name="model"):
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path

   