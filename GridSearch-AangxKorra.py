import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPool1D, Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()

    model.add(Embedding(input_dim = vocab_size,
                        output_dim = embedding_dim,
                        input_length = maxlen,
                        trainable = True))
    model.add(Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation='relu'))

    model.add(Dense(1, activation='sigmoid')) # Output layer

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def word_count_range(df, wc_min, wc_max):
    '''
    Returns the dataframe object with word_count between wc_min and wc_max (exclusive)
    '''
    return df_model[(df_model['word_count'] < wc_max)
                    & (df_model['word_count'] > wc_min)]


with open('./data/df_lines.data', 'rb') as filehandle:
    df_lines = pickle.load(filehandle)

with open('./data/df_lines_korra.data', 'rb') as filehandle:
    df_lines_korra = pickle.load(filehandle)

df_model = pd.concat([
    df_lines[(df_lines['Character'] == 'Aang')][[
        'Character', 'script', 'script_clean', 'word_count'
    ]], df_lines_korra[(df_lines_korra['Character'] == 'Korra')][[
        'Character', 'script', 'script_clean', 'word_count'
    ]]
],
                     ignore_index=True)

# Settings
epochs = 20
embedding_dim = 50
maxlen = 200
output_file = 'data/output-AangxKorra.txt'

# Run grid search
# Get the trimmed version
df_trimmed = word_count_range(df_model, 5, 150)

X = df_trimmed['script_clean'].values
y = df_trimmed['Character'].values
# X = df_model['script'].values
# y = df_model['Character'].values

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# Getting class weights
y_integers = np.argmax(dummy_y, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.2, random_state=101, stratify=dummy_y)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)

# Getting vocab size
vocab_size = len(tokenizer.word_index) + 1 # Adding one because of 0 index

# Padding sequences with zeros
X_train_token = pad_sequences(X_train_token, padding='post', maxlen=maxlen)
X_test_token = pad_sequences(X_test_token, padding='post', maxlen=maxlen)

# Parameter grid for grid search
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  maxlen=[maxlen])

# Build model
model = KerasClassifier(build_fn = create_model,
                        epochs = epochs, verbose = True)

# Grid search
grid = RandomizedSearchCV(estimator = model,
                          param_distributions = param_grid,
                          cv = 4, verbose = 1, n_iter = 5)
grid_result = grid.fit(X_train_token, y_train)

# Evaluate test set
test_accuracy = grid.score(X_test_token, y_test)

with open(output_file, 'a') as f:
        s = ('Best Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
        print(output_string)
        f.write(output_string)
