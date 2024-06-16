import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

def main():
    #lê os arquivos
    df = pd.read_csv('../Dados/asteroides.csv')

    #separa em x as variaveis e em y se isso faz ser perigoso ou não
    x = df.drop('perigo', axis=1)
    y = df['perigo']

    #separa em grupo de teste, para ver a eficacia do classificação
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    print("Tamanho do conjunto de treino:", x_train.shape)
    print("Tamanho do conjunto de teste:", x_test.shape)
    #MLP(x_train, x_test, y_train, y_test)
    #arvore_decicao(x_train, x_test, y_train, y_test)
    #knn(x_train, x_test, y_train, y_test)
    #svc(x_train, x_test, y_train, y_test)
    
if __name__ == "__main__":
    main()
    