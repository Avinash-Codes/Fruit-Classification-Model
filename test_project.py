import pytest
from project import preprocess_image, load_class_names, predict_image_class
import numpy as np
import onnxruntime as ort

# Test for the load_class_names function
def test_load_class_names():
    class_names = load_class_names('class_names.txt')
    assert isinstance(class_names, list)
    assert len(class_names) > 0

# Test for the preprocess_image function
def test_preprocess_image():
    image_path = '31.jpg'
    processed_image = preprocess_image(image_path)
    assert processed_image.shape == (1, 3, 224, 224)

# Test for the predict_image_class function
def test_predict_image_class():
    image = np.random.rand(1, 3, 224, 224).astype(np.float32)
    onnx_model_path = 'fruit_classifier_model.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)
    probabilities = predict_image_class(onnx_model_path, image)
    assert probabilities.shape[1] > 0
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-5)
