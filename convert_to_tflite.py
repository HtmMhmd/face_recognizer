from Model.FaceNet.Facenet import *
import tensorflow as tf

facenet_client = FaceNet512dClient()
model = facenet_client.model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open("Model/FaceNet/facenet.tflite", "wb").write(tflite_model)

