import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
<<<<<<< Updated upstream

def arvore_decicao(x_train, x_test, y_train, y_test):
    #cia a arvore de decicao
        #padrão
    clf = tree.DecisionTreeClassifier()
    
        #colocando os valores padrão
    # clf = tree.DecisionTreeClassifier(
    #     criterion='gini',      # Usa ganho de informação como critério
    #     splitter='best',
    #     max_depth=None,             # Define a profundidade máxima da árvore
    #     min_samples_split=2 ,      # Define o número mínimo de amostras para dividir um nó
    #     min_samples_leaf=1,       # Define o número mínimo de amostras em um nó folha
    #     max_features=None,      # Considera a raiz quadrada do número total de recursos para cada divisão
    #     random_state=None           # Controla a aleatoriedade para reproducibilidade
    # )
=======
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
def arvore_decicao(x_train, x_test, y_train, y_test):
    #cria a arvore de decicao
        
        #padrão
    # clf = tree.DecisionTreeClassifier()
    
        # colocando os valores maior acurracy
    # clf = tree.DecisionTreeClassifier(
    #         ccp_alpha= 0.01,
    #         criterion= 'entropy',
    #         max_depth= None,
    #         max_features= None,
    #         max_leaf_nodes= None,
    #         min_impurity_decrease= 0.0,
    #         min_samples_leaf= 1,
    #         min_samples_split= 2,
    #         min_weight_fraction_leaf= 0.0,
    #         splitter= 'best',
    #         random_state=42,
    #         class_weight='balanced'
    # )
         
         
    clf = tree.DecisionTreeClassifier(
        criterion= 'entropy',
        max_depth= 40, 
        min_samples_leaf= 1,
        min_samples_split= 2, 
        splitter= 'best',
        #random_state=42,
        #class_weight='balanced'
)
         #colocando maior valor de perigo
    # clf = tree.DecisionTreeClassifier(
    #         criterion= 'entropy', 
    #         ccp_alpha= 0.0, 
    #         max_depth= None, 
    #         max_features= None, 
    #         max_leaf_nodes= None, 
    #         min_impurity_decrease= 0.0, 
    #         min_samples_leaf= 1, 
    #         min_samples_split= 2, 
    #         min_weight_fraction_leaf= 0.0, 
    #         splitter= 'best',
    #         random_state=42,
    #         class_weight='balanced'
    # )

#     #treina a arvore
#     clf.fit(x_train,y_train)
>>>>>>> Stashed changes
    
        #mexendo valores
    # clf = tree.DecisionTreeClassifier(
    #     criterion='gini',      # Usa ganho de informação como critério
    #     splitter='best',
    #     max_depth=None,             # Define a profundidade máxima da árvore
    #     min_samples_split=3 ,      # Define o número mínimo de amostras para dividir um nó
    #     min_samples_leaf=1,       # Define o número mínimo de amostras em um nó folha
    #     max_features=None,      # Considera a raiz quadrada do número total de recursos para cada divisão
    #     #class_weight='balanced',    # Ajusta os pesos das classes automaticamente
    #     random_state=42     # Controla a aleatoriedade para reproducibilidade
               
    # )
    
    #treina a arvore
    clf.fit(x_train, y_train)
    
<<<<<<< Updated upstream
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
=======
#     #faz a matriz de confusão
#     cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
#     disp.plot()
#     plt.show()

# def arvore_decicao(x_train, x_test, y_train, y_test, scoring='recall'):
#     # Inicializar um DecisionTreeClassifier
#     clf = tree.DecisionTreeClassifier(random_state=42, class_weight='balanced')

#     # Definir o grid de parâmetros
#     param_grid = {
#         'criterion': ['gini', 'entropy', 'log_loss'],
#         'splitter': ['best'],
#         'max_depth': [None, 5, 10, 20],
#         'min_samples_split': [2, 10, 20],
#         'min_samples_leaf': [1, 5, 10],
#         'min_weight_fraction_leaf': [0.0, 0.1],
#         'max_features': [None, 'sqrt', 'log2'],
#         'max_leaf_nodes': [None, 10],
#         'min_impurity_decrease': [0.0, 0.01],
#         'ccp_alpha': [0.0, 0.01, 0.1]
#     }

#     # Realizar busca em grade com validação cruzada
#     grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=2)
#     grid_search.fit(x_train, y_train)

#     # Melhor combinação de hiperparâmetros
#     print("Melhores parâmetros encontrados: ", grid_search.best_params_)

#     # Treinar o modelo com os melhores parâmetros
#     best_clf = grid_search.best_estimator_
#     best_clf.fit(x_train, y_train)

#     # Fazer previsões no conjunto de teste
#     y_pred = best_clf.predict(x_test)

#     # Imprimir o relatório de classificação
#     target_names = ['não perigoso', 'perigoso']
#     print(classification_report(y_test, y_pred, target_names=target_names))

#     # Matriz de confusão
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
#     disp.plot()
#     plt.show()
>>>>>>> Stashed changes

def MLP(x_train, x_test, y_train, y_test):
    # Crie o classificador MLP
    
    mlp = MLPClassifier()
    # hidden_layer_sizes=(512, 256, 128),  # Aumenta o número de neurônios e camadas ocultas
    # activation='relu',                   # Função de ativação
    # solver='adam',                       # Algoritmo otimizador
    # alpha=0.0001,                        # Regularização L2
    # batch_size=256,                      # Aumenta o tamanho dos lotes
    # learning_rate='adaptive',            # Taxa de aprendizado adaptativa
    # learning_rate_init=0.001,            # Taxa de aprendizado inicial
    # max_iter=1000,                       # Aumenta o número máximo de iterações
    # shuffle=True,                        # Embaralha os dados a cada iteração
    # random_state=42,                     # Semente para reprodução dos resultados
    # tol=1e-6,                            # Reduz a tolerância para otimização
    # verbose=False,                       # Imprime progresso durante a otimização
    # warm_start=False,                    # Reutiliza solução anterior
    # momentum=0.9,                        # Taxa de momento
    # nesterovs_momentum=False,             # Usa momento de Nesterov
    # early_stopping=True,                 # Parada antecipada
    # validation_fraction=0.2,             # Fração de dados para validação
    # beta_1=0.9,                          # Taxa de decaimento exponencial para o primeiro momento
    # beta_2=0.999,                        # Taxa de decaimento exponencial para o segundo momento
    # epsilon=1e-08,                       # Valor para evitar divisão por zero
    # n_iter_no_change=20,                 # Número de iterações sem melhora
    # max_fun=30000                        # Número máximo de chamadas de função


    # Treine o classificador
    mlp.fit(x_train, y_train)

    #Faça previsões no conjunto de teste
    y_pred = mlp.predict(x_test)

    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()


def knn(x_train, x_test, y_train, y_test):
    
    knn = KNeighborsClassifier()
    # Treine o classificador
    knn.fit(x_train, y_train)

    #Faça previsões no conjunto de teste
    y_pred = knn.predict(x_test)

    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

def svc(x_train, x_test, y_train, y_test):
    
    svc = SVC()
    # Treine o classificador
    svc.fit(x_train, y_train)

    #Faça previsões no conjunto de teste
    y_pred = svc.predict(x_test)

    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=svc.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

def naive(x_train, x_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    #Faça previsões no conjunto de teste
    y_pred = clf.predict(x_test)

    #faz o relatório de métricas
    target_names = ['não perigoso', 'perigoso']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    #faz a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['não perigoso', 'perigoso'])
    disp.plot()
    plt.show()

def main():
    #lê os arquivos
    df = pd.read_csv('../Dados/asteroides.csv')

    #separa em x as variaveis e em y se isso faz ser perigoso ou não
    x = df.drop('perigo', axis=1)
    y = df['perigo']

    #separa em grupo de teste, para ver a eficacia do classificação
<<<<<<< Updated upstream
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print("Tamanho do conjunto de treino:", x_train.shape)
    print("Tamanho do conjunto de teste:", x_test.shape)
    #MLP(x_train, x_test, y_train, y_test)
    #arvore_decicao(x_train, x_test, y_train, y_test)
=======
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # #     # Aplicar SMOTE para lidar com desbalanceamento
    # smote = SMOTE(random_state=42)
    # x_train, y_train = smote.fit_resample(x_train, y_train)

    # Normalizar os dados
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #print("Tamanho do conjunto de treino:", x_train.shape)
    #print("Tamanho do conjunto de teste:", x_test.shape)
    MLP(x_train, x_test, y_train, y_test)
    #arvore_decicao(x_train, x_test, y_train, y_test, scoring='recall')
>>>>>>> Stashed changes
    #knn(x_train, x_test, y_train, y_test)
    #svc(x_train, x_test, y_train, y_test)
    naive(x_train, x_test, y_train, y_test)
    
if __name__ == "__main__":
    main()
    