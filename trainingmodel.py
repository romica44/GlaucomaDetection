import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

def entrenar_modelo():
    print("Entrenando el modelo con validación cruzada...")

    path_training = "Database/Glaucoma_Training"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    data_generator = datagen.flow_from_directory(
        path_training,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
        )
    
    # Cargar todas las imágenes en memoria
    X, y = [], []
    num_batches = len(data_generator)
    for i in range(num_batches):
        batch = data_generator[i]
        X.extend(batch[0])
        y.extend(batch[1])

    X = np.array(X)
    y = np.array(y)

    # Crear y compilar el modelo
    def crear_modelo():
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Regularización
            layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Validación cruzada
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    accuracies = []
    losses = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for train, test in kfold.split(X, y):
        model = crear_modelo()
        history = model.fit(X[train], y[train], epochs=20, batch_size=32, verbose=1, validation_data=(X[test], y[test]))

        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}")
        accuracies.append(scores[1])
        losses.append(scores[0])

        y_pred = model.predict(X[test])
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y[test], axis=1)

        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        auc = roc_auc_score(y[test], y_pred, multi_class='ovr')

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Precision for fold {fold_no}: {precision}")
        print(f"Recall for fold {fold_no}: {recall}")
        print(f"F1-Score for fold {fold_no}: {f1}")
        print(f"AUC for fold {fold_no}: {auc}")

        fold_no += 1

    print(f"Average Accuracy: {np.mean(accuracies)}")
    print(f"Average Loss: {np.mean(losses)}")
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1-Score: {np.mean(f1s)}")
    print(f"Average AUC: {np.mean(aucs)}")

    model.save('my_model.h5')

def evaluar_modelo(model_path):
    print("Evaluando el modelo...")

    path_validation = "Database/Glaucoma_Validacion"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    val_ds = datagen.flow_from_directory(
        path_validation,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(val_ds)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    
    # Predicciones y etiquetas verdaderas
    y_pred = model.predict(val_ds)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = val_ds.classes

    # Calcular matriz de confusión
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Matriz de Confusión:")
    print(cm)

    # Calcular precisión, recall, F1-score y ROC AUC
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    y_pred_prob = model.predict(val_ds)
    y_true = val_ds.classes
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])  # Suponiendo que la clase positiva es la segunda

    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
    print(f"Confusion Matrix:\n {cm}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    return test_acc, test_loss, cm, precision, recall, f1, roc_auc

def predecir_con_modelo_entrenado(model_path, path_image):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(path_image, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Expande las dimensiones para que coincida con el formato de entrada del modelo

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1).item()
    print("Predicted class:", predicted_class)
    return predicted_class
