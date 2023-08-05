# By AI-MHT

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Charger l'ensemble de données depuis le fichier CSV
df = pd.read_csv('commentaires.csv')

# Prétraiter les données
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
X = pad_sequences(sequences, maxlen=100)

# Préparer les étiquettes
labels = df['tonality'].astype('category').cat.codes
num_classes = len(labels.unique())
y = tf.keras.utils.to_categorical(labels, num_classes)

# Construire le modèle
inputs = Input(shape=(100,))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Boucle pour permettre la saisie de commentaires multiples
while True:
    # Obtenir l'entrée de l'utilisateur
    test_comment = input("Enter a comment ('q' to quit): ")

    # Vérifier si l'utilisateur souhaite quitter
    if test_comment.lower() == 'q':
        break

    # Prétraiter le commentaire de test
    test_sequence = tokenizer.texts_to_sequences([test_comment])
    test_X = pad_sequences(test_sequence, maxlen=100)

    # Faire une prédiction
    prediction = model.predict(test_X)[0]
    predicted_class = np.argmax(prediction)
    tonalite_classes = df['tonality'].astype('category').cat.categories.tolist()
    predicted_tonalite = tonalite_classes[predicted_class]

    # Afficher la prédiction
    print("The comment is categorized as:", predicted_tonalite)
