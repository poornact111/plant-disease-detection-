import argparse
from plant_disease_classifier import PlantDiseaseClassifier

def train_model(dataset_path):
    # Initialize classifier
    classifier = PlantDiseaseClassifier()
    
    # Prepare data
    print("Preparing data...")
    classifier.prepare_data(dataset_path)
    
    # Build and train model
    print("Training model...")
    history = classifier.train(epochs=10)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Save model
    classifier.save_model('plant_disease_model.h5')
    print("Training completed!")

def predict_disease(image_path):
    # Initialize classifier and load trained model
    classifier = PlantDiseaseClassifier()
    classifier.load_model('plant_disease_model.h5')
    
    # Make prediction
    result = classifier.predict(image_path)
    
    # Display results
    print("\nPlant Disease Analysis Results:")
    print(f"Plant Type: {result['plant_type']}")
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Health Status: {'Healthy' if result['is_healthy'] else 'Diseased'}")

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Classification')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode: train the model or predict on new image')
    parser.add_argument('--dataset', help='Path to the dataset directory (required for train mode)')
    parser.add_argument('--image', help='Path to the plant image (required for predict mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.dataset:
            parser.error("--dataset is required for train mode")
        train_model(args.dataset)
    elif args.mode == 'predict':
        if not args.image:
            parser.error("--image is required for predict mode")
        predict_disease(args.image)

if __name__ == "__main__":
    main()