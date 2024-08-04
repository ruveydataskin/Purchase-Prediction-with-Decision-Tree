import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os


def load_data(file_path):
    """
    Veriyi CSV dosyasından okur ve pandas DataFrame olarak döner.
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Veriyi ön işler ve modelin eğitimine uygun hale getirir.
    Cinsiyet sütununu one-hot encoding yapar.
    """
    X = data[['Cinsiyet', 'Yas', 'TahminiMaas']]
    y = data['SatinAldiMi']
    X = pd.get_dummies(X, columns=['Cinsiyet'], drop_first=True)  # drop_first=True yaparak tek bir dummy değişken
    return X, y


def main():
    # Proje kök dizini
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Veri dosyasının tam yolu
    data_path = os.path.join(base_dir, 'Data/data.csv')

    # Veriyi yükle
    data = load_data(data_path)

    # Veriyi ön işle
    X, y = preprocess_data(data)

    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli eğit
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Test verisi ile tahmin yap
    y_pred = model.predict(X_test)

    # Performansı değerlendir
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Tahmin yapmak istediğimiz veriler
    test_data = pd.DataFrame([[1, 53, 72000]], columns=['Cinsiyet', 'Yas', 'TahminiMaas'])

    # Test verisini aynı şekilde ön işle
    test_data = pd.get_dummies(test_data, columns=['Cinsiyet'], drop_first=True)

    # Eğitim verileriyle uyumlu sütunları oluştur
    test_data = test_data.reindex(columns=X.columns, fill_value=0)

    # Tahmin yap
    prediction = model.predict(test_data)

    # Sonucu yazdır
    if prediction[0] == 0:
        print("Müşteri muhtemelen satın almayacak.")
    else:
        print("Müşteri muhtemelen satın alacak.")


if __name__ == "__main__":
    main()
