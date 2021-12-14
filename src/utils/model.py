import tensorflow as tf
import os
import logging
import keras



def resnet(input_shape,model_path):
    model= tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,weights="imagenet",include_top=False)
    model.save(model_path)
    logging.info(f"resnet50 base model saved at:{model_path}")
    return model