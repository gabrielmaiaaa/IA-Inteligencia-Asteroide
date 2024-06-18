import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


def Meu():
    # CARREGAMENTO DOS DADOS DO ARQUIVO .csv
    df = pd.read_csv('Dados/asteroides.csv')

    # VARIÁVEIS USADAS COMO SEPARAÇÃO DOS VALORES DO ARQUIVO QUE QUEREMOS
    # Váriavel que contém dados do arquivo que vai até a penúltima coluna
    x_test = df.iloc[:, :-1].values
    # Váriavel que contém dados da última coluna do arquivo
    y_test = df.iloc[:, -1].values

    # Separando o treino e validação do nosso dado
    # train_size é usado para distribuir os dados entre a variável de treinamento e de validação (no nosso caso distribui em 80%)
    # random_state pelo que vi funciona para que diferentes pessoas que rodem o código consigam ter os mesmos resultados
    x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size=0.3)

    # CRIANDO O MODELO
    # max_depth recebe a quantidade de nós q vai ter a árvore
    # random_state pelo que vi funciona para que diferentes pessoas que rodem o código consigam ter os mesmos resultados
    clf = DecisionTreeClassifier(random_state=44)

    # TREINANDO O MODELO
    clf.fit(x_train, y_train)

    # Visualizando modelo treinado
    # plot_tree(clf, filled=True)
    # plt.show()

    # VALIDANDO O MODELO
    y_predict = clf.predict(x_val)

    # Calcular a acurácia
    accuracy = accuracy_score(y_val, y_predict)
    print(f'Acurácia: {accuracy:.2f}')

    # AVALIANDO DESEMPENHO DO MODELO
    cm = confusion_matrix(y_val, y_predict)
    print(classification_report(y_val, y_predict))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

def knn():
    # CARREGAMENTO DOS DADOS DO ARQUIVO .csv
    df = pd.read_csv('Dados/asteroides.csv')

    # VARIÁVEIS USADAS COMO SEPARAÇÃO DOS VALORES DO ARQUIVO QUE QUEREMOS
    # Váriavel que contém dados do arquivo que vai até a penúltima coluna
    x = df.iloc[:, :-1].values
    # Váriavel que contém dados da última coluna do arquivo
    y = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)
    
    # Aplicar SMOTE para lidar com desbalanceamento
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Normalizar os dados
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Treinar KNN com melhor parâmetro
    KNN = KNeighborsClassifier(n_neighbors=10)
    KNN.fit(x_train, y_train)

    # VALIDANDO O MODELO
    y_pred = KNN.predict(x_test)

    # Nossa classificação do modelo
    print(classification_report(y_test, y_pred, target_names=['não perigoso', 'perigoso']))
    
    # Plot o nosso modelo
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

knn()