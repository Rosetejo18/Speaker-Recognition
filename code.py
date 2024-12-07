import librosa
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define functions
def mfcc_avg(files):
    avg = []
    for file in files:
        samples, samplingrate = librosa.load(file, sr=16000, mono=True, offset=0.0, duration=None)
        mfcc = librosa.feature.mfcc(y=samples, sr=samplingrate, win_length=400, hop_length=200, n_mfcc=13)
        avg.append(np.average(mfcc, axis=1))
    return np.average(avg, axis=0)

def distance(avg, test, cov_inv):
    return mahalanobis(avg, test, cov_inv)

def predict_speaker(test_file, avg_i, cov_inv):
    samples, samplingrate = librosa.load(test_file, sr=16000, mono=True, offset=0.0, duration=None)
    mfcc_test = librosa.feature.mfcc(y=samples, sr=samplingrate, win_length=400, hop_length=200, n_mfcc=13)
    avg_test = np.average(mfcc_test, axis=1)
    
    distances = {speaker: distance(avg_i[speaker], avg_test, cov_inv) for speaker in avg_i}
    predicted_speaker = min(distances, key=distances.get)
    
    return predicted_speaker

# Dataset
dataset={
         'Sandhya':["vsa1.wav", "vsa2.wav", "vsa3.wav"],
         'Rose':["vro1.wav", "vro2.wav", "vro3.wav"],
         'Adithya':["vak1.wav", "vak2.wav", "vak3.wav"],
         'Nirupama':["vni1.wav","vni2.wav","vni3.wav"],
         'Jayanth':["vja1.wav","vja2.wav","vja3.wav"],
         }

# Calculate average of MFCC features
avg_i = {}
for ind in dataset:
    avg_i[ind] = mfcc_avg(dataset[ind])

# Calculate covariance matrix and its inverse for Mahalanobis distance
stack = np.vstack(list(avg_i.values()))
covm = np.cov(stack.T)
cov_inv = np.linalg.pinv(covm)

# Test dataset with true labels
test_dataset = {
    'chkni.wav': 'Nirupama',
    'chkro.wav': 'Rose',
    'chksa.wav': 'Sandhya',
    'chkja.wav': 'Jayanth',
    'chkak.wav': 'Adithya'
}

#chatgpt code to calc metrics
# Collect true labels and predictions
true_labels = []
predicted_labels = []

for test_file, true_speaker in test_dataset.items():
    true_labels.append(true_speaker)
    predicted_labels.append(predict_speaker(test_file, avg_i, cov_inv))

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
