import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Path to directory containing CSV files
df = pd.read_csv('tennis.csv')

keep_cols = ['ATP', 'Location', 'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets', 'MaxW', 'MaxL', 'AvgW', 'AvgL']

# Drop all columns except the ones in keep_cols
df = df[keep_cols]

df['Location'] = pd.factorize(df['Location'])[0]
df['Tournament'] = pd.factorize(df['Tournament'])[0]
df['Series'] = pd.factorize(df['Series'])[0]
df['Court'] = pd.factorize(df['Court'])[0]
df['Surface'] = pd.factorize(df['Surface'])[0]
df['Round'] = pd.factorize(df['Round'])[0]

df['RankDiff'] = df['WRank'] - df['LRank']

print(df.dtypes)

X = df.drop(['Winner'], axis=1)
y = df['Winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()

# Build neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model on test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.3f}')

