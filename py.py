import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to extract features from audio files
def extract_features(file_name):
	try:
		audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
		mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
		mfccs_scaled = np.mean(mfccs.T, axis=0)
		return mfccs_scaled
	except Exception as e:
		print(f"Error encountered while parsing file: {file_name}")
		print(e)
		return None

# Load dataset and extract features
def load_data(data_dir):
	emotions = []
	features = []
	for emotion_dir in os.listdir(data_dir):
		emotion_path = os.path.join(data_dir, emotion_dir)
		if os.path.isdir(emotion_path):
			for file_name in os.listdir(emotion_path):
				if file_name.endswith('.wav'):
					file_path = os.path.join(emotion_path, file_name)
					feature = extract_features(file_path)
					if feature is not None:
						features.append(feature)
						emotions.append(emotion_dir)
					else:
						print(f"Failed to extract features from {file_path}")
	return np.array(features), np.array(emotions)

# Path to the TESS Toronto emotional speech set data
data_dir = r'C:\Benish\TESS Toronto emotional speech set data'

# Verify the dataset structure
print("Verifying dataset structure...")
if not os.path.exists(data_dir):
	print(f"Data directory {data_dir} does not exist.")
else:
	for emotion_dir in os.listdir(data_dir):
		emotion_path = os.path.join(data_dir, emotion_dir)
		if os.path.isdir(emotion_path):
			print(f"Found emotion directory: {emotion_dir}")
			for file_name in os.listdir(emotion_path):
				if file_name.endswith('.wav'):
					print(f"Found audio file: {file_name}")
		else:
			print(f"Skipping non-directory item: {emotion_dir}")

# Load the data
print("Loading data...")
X, y = load_data(data_dir)
print(f"Loaded {len(X)} samples.")

# Check if data is loaded correctly
if len(X) == 0:
	print("No data loaded. Please check the data directory path and file formats.")
else:
	# Split the dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train the model
	print("Training the model...")
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)

	# Evaluate the model
	print("Evaluating the model...")
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f'Accuracy: {accuracy * 100:.2f}%')
model_filename = 'speech_emotion_detection_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")