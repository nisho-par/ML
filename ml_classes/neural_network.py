from dataset import X_train, y_train, X_test, y_test
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()



import tensorflow as tf

def train_model(X_train, y_train, epochs, batch_size, validation_split,  num_nodes, dropout_prob, learning_rate):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(34,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
        
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', 
                    metrics=['accuracy'])
    
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split
    )
    
    return nn_model, history

nn_model, history = train_model(X_train=X_train, y_train=y_train, epochs=100, batch_size=100, validation_split=0.2, num_nodes=34, dropout_prob=0.05, learning_rate=0.001)

# plot_loss(history=history)
# plot_accuracy(history=history)

y_pred = nn_model.predict(X_test)
# convert the sigmoid function results to 0 and 1
y_pred = (y_pred>0.5).astype(int).reshape(-1,)

report = classification_report(y_test, y_pred)
print(report)