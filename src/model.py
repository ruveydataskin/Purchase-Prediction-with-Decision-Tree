import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os


def load_data(file_path):
    """
    Reads data from a CSV file and returns it as a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Preprocesses the data and prepares it for model training.
    Converts the 'Gender' column to one-hot encoding.
    """
    X = data[['Gender', 'Age', 'EstimatedSalary']]
    y = data['Purchased']
    X = pd.get_dummies(X, columns=['Gender'], drop_first=True)  # drop_first=True to avoid dummy variable trap
    return X, y


def main():
    # Project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Full path to the data file
    data_path = os.path.join(base_dir, 'Data/data.csv')

    # Load the data
    data = load_data(data_path)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the performance
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Data for making predictions
    test_data = pd.DataFrame([[1, 59, 88000]], columns=['Gender', 'Age', 'EstimatedSalary'])

    # Preprocess the test data in the same way
    test_data = pd.get_dummies(test_data, columns=['Gender'], drop_first=True)

    # Align test data columns with training data columns
    test_data = test_data.reindex(columns=X.columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(test_data)

    # Print the result
    if prediction[0] == 0:
        print("The customer is unlikely to make a purchase.")
    else:
        print("The customer is likely to make a purchase.")


if __name__ == "__main__":
    main()
