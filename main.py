import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the CSV file into a pandas DataFrame
tennis_data = pd.read_csv('data/2000.csv')

# Drop unnecessary columns
tennis_data.drop(['Comment'], axis=1, inplace=True)

# Handle missing values
tennis_data = tennis_data.replace('NR', np.nan)
tennis_data.fillna(0, inplace=True)

# Encode categorical features
tennis_data = pd.get_dummies(tennis_data, columns=['Location', 'Tournament', 'Series', 'Court', 'Surface'])


print(tennis_data.head().to_string())

# Create new features
tennis_data['Rank_diff'] = tennis_data['WRank'].astype(int) - tennis_data['LRank'].astype(int)
tennis_data['Sets_diff'] = tennis_data['Wsets'].astype(int) - tennis_data['Lsets'].astype(int)



# Split the data into training and test sets
X = tennis_data.drop(['Winner', 'Loser'], axis=1)
y = tennis_data['Winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate the model
y_pred = model.predict_classes(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

