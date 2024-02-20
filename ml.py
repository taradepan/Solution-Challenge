import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


def train_model():
    df = pd.read_csv("./parkinsons.data")

    # Separate features and target
    X = df.drop(columns=['name', 'status'], axis=1)
    Y = df['status']

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Scale the features
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    # Train the SVM model
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    # Evaluate the model
    Y_train_pred = model.predict(X_train)
    train_data_acc = accuracy_score(Y_train, Y_train_pred)
    print("Accuracy of training data:", train_data_acc)

    Y_test_pred = model.predict(X_test)
    test_data_acc = accuracy_score(Y_test, Y_test_pred)
    print('Accuracy of testing data:', test_data_acc)

    # Provide input data for prediction
    input_data = (120.25600, 125.30600, 104.77300, 0.00407, 0.00003, 0.00224, 0.00205, 0.00671, 0.01516,
                0.13800, 0.00721, 0.00815, 0.01310, 0.02164, 0.01015, 26.01700, 1, 0.468621, 0.735136,
                -6.112667, 0.217013, 2.527742)  # Remove the last feature
    input_data_np = np.asarray(input_data)
    input_data_re = input_data_np.reshape(1, -1)

    # Scale the input data using the trained scaler
    s_data = ss.transform(input_data_re)

    # Make predictions
    pred = model.predict(s_data)
    print("Prediction:", pred)

    # Output result
    if pred[0] == 0:
        print("Negative, No Parkinson's")
    else:
        print("Positive, Parkinson's")

    pickle.dump(model,open('model.pkl', 'wb'))
