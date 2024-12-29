from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the dataset
bcancer = datasets.load_breast_cancer()
X = bcancer.data
y = bcancer.target

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape target for compatibility with OneHotEncoder
y = y.reshape(-1, 1)

# Split into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=11)

# One-hot encode the target
oh = OneHotEncoder()
ytrain = oh.fit_transform(ytrain).toarray()
ytestoh = oh.transform(ytest).toarray()  # Use transform, not fit_transform

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))  # Match input_dim to number of features
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Output layer with 2 units for binary classification

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(Xtrain, ytrain, epochs=200, batch_size=100, validation_data=(Xtest, ytestoh))

# Predict and evaluate
ypred = model.predict(Xtest)
ypred = np.argmax(ypred, axis=1)
ytrue = np.argmax(ytestoh, axis=1)  # Convert one-hot to class labels for comparison

# Accuracy score
score = accuracy_score(ypred, ytrue)
print(f'Accuracy score is {100 * score:.2f}%')

# Confusion matrix
cmat = confusion_matrix(ytrue, ypred)
print('Confusion matrix of Neural Network is \n', cmat, '\n')

# Plot training and validation loss
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'g-', linewidth=3, label='Training Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], 'g-.', linewidth=3, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.title('Loss vs. Epochs')
plt.show()

# Plot training and validation accuracy
plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], 'b-', linewidth=3, label='Training Accuracy')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], 'b-.', linewidth=3, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.show()

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=bcancer.target_names)
disp.plot()
plt.show()
