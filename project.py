import onnxruntime as ort
import numpy as np
from scipy.special import softmax
from torchvision import transforms
from PIL import Image

# Define the image transformations
def preprocess_image(image_path, input_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).numpy()  # Add batch dimension and convert to NumPy array

# Load class names from file
def load_class_names(file_path='class_names.txt'):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Run the ONNX model and get predictions
def predict_image_class(onnx_model_path, image):
    ort_session = ort.InferenceSession(onnx_model_path)
    inputs = {ort_session.get_inputs()[0].name: image}
    outputs = ort_session.run(None, inputs)
    return softmax(outputs[0], axis=1)

# Main function to run the prediction and display results
def main():
    image_path = input("Enter the path: ")  # Update with your image path
    onnx_model_path = 'fruit_classifier_model.onnx'

    # Load class names
    class_names = load_class_names()

    # Preprocess image
    image = preprocess_image(image_path)

    # Run prediction
    probabilities = predict_image_class(onnx_model_path, image)
    predicted_idx = np.argmax(probabilities, axis=1)

    # Get predicted class and probability
    predicted_class = class_names[predicted_idx[0]]
    predicted_probability = probabilities[0][predicted_idx[0]] * 100

    # Get and print top 5 probabilities
    probabilities_list = {class_names[i]: probabilities[0][i] * 100 for i in range(len(class_names))}
    top_5_classes = sorted(probabilities_list.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f'The predicted fruit is: {predicted_class} with a probability of {predicted_probability:.2f}%')
    print("\nTop 5 classes with highest probabilities:")
    for fruit, prob in top_5_classes:
        print(f"{fruit}: {prob:.2f}%")

if __name__ == "__main__":
    main()
