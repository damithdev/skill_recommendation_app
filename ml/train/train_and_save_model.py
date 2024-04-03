# import pandas as pd
# from ml.text_classifier import TextClassifier
# import joblib
# from config import MODEL_FILE_PATH
#
# # Read the training data from the CSV file
# data = pd.read_csv('path/to/training_data.csv')
# X = data['text'].tolist()
# y = data['label'].tolist()
#
# # Train the classifier
# classifier = TextClassifier()
# classifier.train(X, y)
#
# # Save the trained ml
# joblib.dump(classifier.model, MODEL_FILE_PATH)
#
# # Load the trained ml for testing
# loaded_pipeline = joblib.load(MODEL_FILE_PATH)
# loaded_classifier = TextClassifier( model=loaded_pipeline )
#
# # Testing with new examples
# test_samples = ["Deep learning techniques", "JavaScript and web development", "Data analysis in Python"]
# for test in test_samples:
#     prediction = loaded_classifier.predict(test)
#     print(f"Prediction for '{test}': {prediction}")
