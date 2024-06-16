import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=44)

# Criando modelo
# max_depth recebe a quantidade de nós q vai ter a árvore
# random_state pelo que vi funciona para que diferentes pessoas que rodem o código consigam ter os mesmos resultados
clf = DecisionTreeClassifier(max_depth=2, random_state=44)

# Treinando modelo
clf.fit(x_train, y_train)

# Visualizando modelo treinado
plot_tree(clf, filled=True)
plt.show()