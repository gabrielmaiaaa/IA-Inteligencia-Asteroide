import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

def knn_parametros(df):
    # Separar as características e o rótulo
    x = df.drop('perigo', axis=1)
    y = df['perigo']

    # Balancear as classes usando oversampling (SMOTE) ou undersampling
    df_majority = df[df.perigo == 0]
    df_minority = df[df.perigo == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results

    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    x_balanced = df_balanced.drop('perigo', axis=1)
    y_balanced = df_balanced['perigo']

    # Separar em grupo de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=0.1, random_state=42)

    # Normalizar as características
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Definindo a grade de parâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Inicializando o classificador KNN
    knn = KNeighborsClassifier()

    # Criando um scorer customizado para o recall da classe "perigoso"
    scorer = make_scorer(recall_score, pos_label=1)

    # Configurando o GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scorer, n_jobs=-1)

    # Treinando o GridSearchCV
    grid_search.fit(x_train, y_train)

    # Obtendo o melhor classificador
    best_clf = grid_search.best_estimator_
    best_clf.fit(x_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = best_clf.predict(x_test)

    # Imprimindo o relatório de classificação
    target_names = ['não perigoso', 'perigoso']
    print("Tamanho do conjunto de treino:", x_train.shape)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    # Retornando o melhor classificador e os melhores parâmetros
    return best_clf

def knn_semParametro(df):
    # Separar as características e o rótulo
    x = df.drop('perigo', axis=1)
    y = df['perigo']

    # Balancear as classes usando oversampling (SMOTE) ou undersampling
    df_majority = df[df.perigo == 0]
    df_minority = df[df.perigo == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results

    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    x_balanced = df_balanced.drop('perigo', axis=1)
    y_balanced = df_balanced['perigo']

    # Separar em grupo de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=0.1, random_state=42)

    # Normalizar as características
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = KNeighborsClassifier()

    #treina a arvore
    clf.fit(x_train, y_train)
    
    #faz o predict das amostras que a gente deixou como teste
    y_pred = clf.predict(x_test)
    
    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

def knn_parametro_accuracy(df):
    # Separar as características e o rótulo
    x = df.drop('perigo', axis=1)
    y = df['perigo']

    # Balancear as classes usando oversampling
    df_majority = df[df.perigo == 0]
    df_minority = df[df.perigo == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results

    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    x_balanced = df_balanced.drop('perigo', axis=1)
    y_balanced = df_balanced['perigo']

    # Separar em grupo de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=0.1, random_state=42)

    # Normalizar as características
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Definindo a grade de parâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Inicializando o classificador KNN
    knn = KNeighborsClassifier()

    # Configurando o GridSearchCV para maximizar a acurácia
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Treinando o GridSearchCV
    grid_search.fit(x_train, y_train)

    # Obtendo o melhor classificador
    best_clf = grid_search.best_estimator_
    best_clf.fit(x_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = best_clf.predict(x_test)

    # Imprimindo o relatório de classificação
    target_names = ['não perigoso', 'perigoso']
    print("Tamanho do conjunto de treino:", x_train.shape)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    # Retornando o melhor classificador e os melhores parâmetros
    return best_clf, grid_search.best_params_

def knn_parametros_sem(df):
      #separa em x as variaveis e em y se isso faz ser perigoso ou não
    x = df.drop('perigo', axis=1)
    y = df['perigo']


    #separa em grupo de teste, para ver a eficacia do classificação
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

     # Definindo a grade de parâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

     # Inicializando o classificador KNN
    knn = KNeighborsClassifier()

    # Criando um scorer customizado para o recall da classe "perigoso"
    scorer = make_scorer(recall_score, pos_label=1)

    # Configurando o GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scorer, n_jobs=-1)

    # Treinando o GridSearchCV
    grid_search.fit(x_train, y_train)

    # Obtendo o melhor classificador
    best_clf = grid_search.best_estimator_
    best_clf.fit(x_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = best_clf.predict(x_test)

    # Imprimindo o relatório de classificação
    target_names = ['não perigoso', 'perigoso']
    print("Tamanho do conjunto de treino:", x_train.shape)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    # Retornando o melhor classificador e os melhores parâmetros
    return best_clf

def knn_semParametro_sem(df):

    x = df.drop('perigo', axis=1)
    y = df['perigo']

    #separa em grupo de teste, para ver a eficacia do classificação
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    
    clf = KNeighborsClassifier()

    #treina a arvore
    clf.fit(x_train, y_train)
    
    #faz o predict das amostras que a gente deixou como teste
    y_pred = clf.predict(x_test)
    
    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()


def knn_parametro_accuracy_sem(df):

    x = df.drop('perigo', axis=1)
    y = df['perigo']

    #separa em grupo de teste, para ver a eficacia do classificação
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


     # Definindo a grade de parâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Inicializando o classificador KNN
    knn = KNeighborsClassifier()

    # Configurando o GridSearchCV para maximizar a acurácia
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Treinando o GridSearchCV
    grid_search.fit(x_train, y_train)

    # Obtendo o melhor classificador
    best_clf = grid_search.best_estimator_
    best_clf.fit(x_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = best_clf.predict(x_test)

    # Imprimindo o relatório de classificação
    target_names = ['não perigoso', 'perigoso']
    print("Tamanho do conjunto de treino:", x_train.shape)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    # Retornando o melhor classificador e os melhores parâmetros
    return best_clf, grid_search.best_params_

def main():
    #lê os arquivos
    df = pd.read_csv('../Dados/asteroides.csv')

    #com tratamento
    #best_clf= knn_parametros(df)
    #print(best_clf)
    #knn_semParametro(df)
    #best_clf, grid = knn_parametro_accuracy(df)
    #print(best_clf)

    #sem tratamento
    best_clf = knn_parametros_sem(df)
    print(best_clf)
    knn_semParametro_sem(df)
    best_clf, grid = knn_parametro_accuracy_sem(df)
    print(best_clf)

if __name__ == "__main__":
    main()
