import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib
# Load and preprocess data
df = pd.read_csv("dataset.csv")

# Feature selection
feature_columns = [
    "How often do you feel stressed?",
    "How often do you like hanging out outside campus? (clubbing, cafe hopping etc)",
    "How much you value sense of humour in a friend?",
    "How likely are you to join a new club or society that aligns with your passions and interests?",
    "How much do you like exploring new places? ",
    "How much are you involved in physical activities (sports/gym)?",
    "How much do you like reading?",
    "How much are you into video games?",
    "How much do you like watching movies/series?",
    "How much do you prefer deep conversations or casual banter in friendships?"
]

X = df[feature_columns]
y = df["What's your personality type ? ( 5 being ambivert )"]

# Categorize personality types
def categorize_personality(value):
    if value <= 4:
        return 0  # Introvert
    elif value == 5:
        return 1  # Ambivert
    else:
        return 2  # Extrovert

y_categorized = y.apply(categorize_personality)

# Print class distribution before balancing
print("Class distribution before balancing:")
print(y_categorized.value_counts())

# Simple data balancing
def simple_balance_dataset(X, y):
    # Combine features and target
    data = pd.concat([X, pd.Series(y, name='target')], axis=1)

    # Get minimum class size
    min_class_size = data['target'].value_counts().min()

    # Balance each class to minimum size
    balanced_dfs = []
    for class_label in data['target'].unique():
        class_data = data[data['target'] == class_label]
        if len(class_data) > min_class_size:
            class_data = class_data.sample(n=min_class_size, random_state=42)
        balanced_dfs.append(class_data)

    # Combine balanced data
    balanced_data = pd.concat(balanced_dfs)
    balanced_data = shuffle(balanced_data, random_state=42)

    return balanced_data.drop('target', axis=1), balanced_data['target']

# Balance dataset
X_balanced, y_balanced = simple_balance_dataset(X, y_categorized)

# Print class distribution after balancing
print("\nClass distribution after balancing:")
print(y_balanced.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced  # Ensure balanced split
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simplified model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=0.01
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001
)

# Train the model with a smaller batch size
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=8,  # Smaller batch size
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
joblib.dump(model,'model.joblib')
# Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Function for predictions with confidence
def predict_personality(input_data):
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Get predictions
    predictions = model.predict(input_scaled)

    # Get class probabilities
    class_probs = predictions[0]

    # Get predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map to labels
    class_labels = {0: "Introvert", 1: "Ambivert", 2: "Extrovert"}

    # Calculate confidence
    confidence = class_probs[predicted_class] * 100

    return {
        'predicted_type': class_labels[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'Introvert': class_probs[0] * 100,
            'Ambivert': class_probs[1] * 100,
            'Extrovert': class_probs[2] * 100
        }
    }

# Print training history
print("\nTraining History:")
for epoch, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
    print(f"Epoch {epoch + 1}: Training Accuracy = {acc:.4f}, Validation Accuracy = {val_acc:.4f}")

# Example prediction
example_input = [[1, 10, 1, 10, 10, 2, 2, 2, 9, 10]]
result = predict_personality(example_input)

print("\nPrediction Results:")
print(f"Predicted Personality Type: {result['predicted_type']}")
print(f"Confidence: {result['confidence']:.2f}%")
print("\nProbabilities for each type:")
for personality_type, prob in result['probabilities'].items():
    print(f"{personality_type}: {prob:.2f}%")