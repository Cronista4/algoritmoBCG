import time
import pandas as pd
import scipy
from cffi.backend_ctypes import xrange

import palavrasParada
import preProcessador
import numpy
import sys
import textblob
from googletrans import Translator
from textblob import TextBlob

# generate random integer values
from random import seed
from random import randint

##Classificadores
from sklearn.feature_extraction.text import CountVectorizer  ## Converter string para vetor
from sklearn.feature_extraction import text  # Possui stop-words
from sklearn.feature_extraction.text import TfidfVectorizer  # Extração TFIDF
from sklearn.model_selection import train_test_split  ## Método hold-out
from sklearn.linear_model import LogisticRegression  ## Regressão logística
from sklearn import svm  # SVM
from sklearn.naive_bayes import GaussianNB  # NB
from sklearn.ensemble import RandomForestClassifier  # RF
from sklearn.linear_model import SGDClassifier  # SGD - Stochastic Gradient Descent
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn import tree  # C4,5
from sklearn.neural_network import MLPClassifier  # MLP

##Métricas de qualidade
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score

from ReliefF import ReliefF

def main():

    print("Teste com SKLEARN")

    amostras = 200
    ativaSelecao = False
    estadoInicial = 1
    selecaoAtributosAtiva = True
    k = 1000

    print("\nDados brutos -", amostras, "amostras: \n")
    # dataset_imdb.csv comentarios_CLASME.csv comentarios_CLASME_2classes.csv opcovid_br_en.csv kaggle_sanders_10k.csv base_bruta_2_classes_ingles

    # Base completa: base_bruta_3_classes_cheia_ingles -> 12000 tuplas
    # Base completa: base_limpa_3_classes_cheia_ingles -> 12000 tuplas

    # Base menor: base_bruta_3_classes_ingles -> 6000 tuplas
    # Base rápida: base_rapida_ingles_2_classes -> 200, base_rapida_ingles_3_classes -> 300
    # Base que provavelmente será a final: base_limpa_ingles_3_classes_NOVA_6000 -> 6000
    # Base nova 600 -> base_limpa_ingles_3_classes_NOVA_600

    # Base nova com stop words padrão base_limpa_padrao_ingles_3_classes_NOVA_600 *******
    # Base para testes médios: base_limpa_padrao_ingles_3_classes_NOVA_1200

    # Base nova com stop words e POS_tag base_limpa_padrao_ingles_3_classes_NOVA_POS_TAG_600

    dadosFilmes = pd.read_csv('comentarios/base_limpa_padrao_ingles_3_classes_NOVA_1200.csv')#.sample(amostras,random_state=1)##random_state=estadoInicial
    print(dadosFilmes.sentiment.value_counts())
    print(dadosFilmes.head())

    #comentarios = dadosFilmes['review']
    #print("###")
    #print(comentarios)

    # Troca a classe para valores numéricos
    dadosFilmes.sentiment = dadosFilmes['sentiment'].map({'positiva': 1, 'negativa': -1, 'neutra': 0})
    print("\nAgora os sentimentos estão como classes numéricas:\n")
    print(dadosFilmes)
    ##Tradução dos dados de português para inglês (deve ser feito só uma vez):
    #for frase in dadosFilmes.review:
    #    print("FRASE: ",frase, type(frase))
    #    fraseBlob = TextBlob(frase)
    #    resultado = fraseBlob.translate(from_lang="pt",to="en")
    #    print("RESULTADO: ", resultado, type(resultado))
    #    dadosFilmes['review'] = dadosFilmes['review'].replace(frase, str(resultado) ) #O dataframe recebe uma string = str
    #dadosFilmes.to_csv(r'C:\Users\Luis\Documents\GBD\SVN\Academico\datamining\minha\Sistemas\python\algoritmoGenetico\comentarios\port.csv', index = False)
    #print("verifique")
    #time.sleep(10)
    #print(dadosFilmes.head())
    #print(dadosFilmes.sentiment.head())

    print("\n################################\n")
    #stop_words = text.ENGLISH_STOP_WORDS.union(palavrasParada.minhas_paralavras_de_parada)
    stop_words = palavrasParada.minhas_paralavras_de_parada ##Maior acurácia no M.E.
    print("Palavras de parada usadas: ", stop_words, "\n", len(stop_words))
    # time.sleep(1)
    print("\n################################\n")

    # Converte uma coleção de documentos brutos em uma matriz de características TF-IDF.
    vetor = TfidfVectorizer(smooth_idf=True,  ##Evita divisão por zero na contagem
                            use_idf=True,
                            min_df=3,  # Ignore termos que aparecem em menos de 3 documentos (1)
                            max_df=0.7,  # Ignore termos que aparecem em 70% ou mais dos documentos
                            lowercase=True,  # Converte as letras para minúsculas
                            stop_words=stop_words,  # minha lista de stop-words
                            #preprocessor=preProcessador.meu_preprocessador,
                            ##Realiza processos de remoção caracteres especiais e radicalização

                            ngram_range=(1, 1)  ## N-gramas 1,1 unigrama
                           )



    #print("Vetor antes do FIT",vetor)
    vetor = vetor.fit(dadosFilmes.review)  # Com base nos parâmetros anteriores, aprende um vocabulário a partir das reviews com todos os tokens que sobraram
    #print("Vetor depois do FIT", vetor)

    matrizTexto = vetor.transform(dadosFilmes.review)  # Transform documents to document-term matrix.
    # Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.

    print("\nVetor de texto (linhas:colunas): ", matrizTexto.shape)
    print("Sentimentos:\n", dadosFilmes.sentiment)

    # print("Vocabulário: \n", vetor.vocabulary_)
    # print("VETOR:\n", vetor)

    print("\n ->", len(vetor.vocabulary_))
    print("matriz Texto:\n", matrizTexto)
    # print("TIPO: ", type(matrizTexto))

    print("ATRIBUTOS: \n", vetor.get_feature_names())

    numpy.set_printoptions(threshold=sys.maxsize)  ##Exibe a matriz por inteiro
    print("MATRIZ: \n", matrizTexto.shape, "\n")
    #print("", matrizTexto.toarray())

    ##ReliefF
    fs = ReliefF(n_neighbors=1, n_features_to_keep=150)
    X_train = fs.fit_transform(matrizTexto, dadosFilmes.review)
    print(X_train)
    print("--------------")
    print("(No. of tuples, No. of Columns before ReliefF) : " + str(matrizTexto.shape) +
          "\n(No. of tuples , No. of Columns after ReliefF) : " + str(X_train.shape))

    time.sleep(1000)

    matrizModificada = pd.DataFrame(matrizTexto.toarray())  # não perder a original, criamos uma matriz densa dataframe
    numpy.set_printoptions(threshold=sys.maxsize)  ##Exibe a matriz por inteiro
    print("ANTES:\n", matrizModificada)
    # matrizModificada=matrizModificada.drop(2,1) -> index 2, 1 = coluna -> coluna 2

    numpy.set_printoptions(threshold=sys.maxsize)  ##Exibe a matriz por inteiro
    print("DEPOIS:\n", matrizModificada)
    matrizModificada = scipy.sparse.csr_matrix(matrizModificada)  # Converte a matrizModificada para esparsa que é o que o classificador precisa
    numpy.set_printoptions(threshold=sys.maxsize)  ##Exibe a matriz por inteiro
    print("Em forma esparsa - DM:\n", matrizModificada)


    ##Separação dos dados - 70% - 30%
    ##Função train_test_split. Precisa receber: matriz de termos, as classes, o tamanho do set de teste e uma seed
    ## Os x são os atributos e Y é a classe
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        # matrizTexto, ##CORRETO
        matrizModificada,
        dadosFilmes.sentiment,
        test_size=0.3,
        random_state=estadoInicial
    )
    #print("\nx_treino\n",x_treino)
    #print("\nx_teste\n",x_teste)
    #print("\ny_treino\n",y_treino)
    #print("\ny_teste\n",y_teste)

    # print("ATRIBUTOS: \n", vetor.get_feature_names())  ##Em ordem alfabética

    ##Classificadores:
    print("#####################################")
    print("\nClassificador regressão logistica - Max Ent:\n")
    tempoExecucao = time.time()
    # solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
    # Algorithm to use in the optimization problem.
    # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
    # ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
    # ‘liblinear’ and ‘saga’ also handle L1 penalty
    # ‘saga’ also supports ‘elasticnet’ penalty
    # ‘liblinear’ does not support setting penalty='none'
    regLog = LogisticRegression(C=2,
                                random_state=0,
                                solver='sag')
    regLog = regLog.fit(x_treino, y_treino)
    y_predito = regLog.predict(x_teste)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    #curvaROC = roc_auc_score(y_predito, y_teste)
    #print("Curva ROC: ", curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)

    print("#####################################")
    print("\nClassificador NB:\n")
    tempoExecucao = time.time()
    bayesianoIngenuo = GaussianNB()
    bayesianoIngenuo.fit(x_treino.toarray(), y_treino)
    y_predito = bayesianoIngenuo.predict(x_teste.toarray())
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    #curvaROC = roc_auc_score(y_predito, y_teste)
    #print("Curva ROC: ", curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)

    print("#####################################")
    print("\nClassificador SVM:\n")
    tempoExecucao = time.time()
    clSVM = svm.SVC(kernel='linear',
                    degree=3,
                    max_iter=-1)  ## os kernels disponíveis são ('linear', 'poly', 'rbf')
    clSVM = clSVM.fit(x_treino, y_treino)
    y_predito = clSVM.predict(x_teste)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    #curvaROC = roc_auc_score(y_predito, y_teste)
    #print("Curva ROC: ", curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)

    print("#####################################")
    print("\nClassificador Random Forest:\n")
    tempoExecucao = time.time()

    clRF = RandomForestClassifier(n_estimators=20, random_state=0)
    clRF = clRF.fit(x_treino, y_treino)
    y_predito = clRF.predict(x_teste)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    #curvaROC = roc_auc_score(y_predito, y_teste)
    #print("Curva ROC: ", curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)

    print("#####################################")
    print("\nClassificador K-NN:\n")
    tempoExecucao = time.time()
    clKNN = KNeighborsClassifier(n_neighbors=20)
    clKNN = clKNN.fit(x_treino, y_treino)
    y_predito = clKNN.predict(x_teste)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    # curvaROC = roc_auc_score(y_predito, y_teste)
    # print("Curva ROC: ",curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)

    print("\nClassificador Árvore de decisão:\n")
    tempoExecucao = time.time()
    clAD = tree.DecisionTreeClassifier()
    clAD = clAD.fit(x_treino, y_treino)
    y_predito = clAD.predict(x_teste)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:", tempoExecucao, "segundos")

    acuracia = accuracy_score(y_predito, y_teste)
    print("Acurácia: ", acuracia)
    precisao = precision_score(y_predito, y_teste, average='weighted')
    print("Precisão: ", precisao)
    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    print("Sensibilidade: ", sensibilidade)
    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    print("Erro absoluto médio: ", erroMedioAbsoluto)
    # curvaROC = roc_auc_score(y_predito, y_teste)
    # print("Curva ROC: ",curvaROC)
    f1 = f1_score(y_predito, y_teste, average='weighted')
    print("F1: ", f1)
    matrizConfusao = confusion_matrix(y_predito, y_teste)
    print("Matriz de confusão: \n", matrizConfusao)


    #k_features = [500, 1000, 2000, 4000, 10000]
    # x_treino, x_teste, y_treino, y_teste
    # A função de seleção atualiza os conjuntos de treino e teste
    # for k in k_features:

    # print("select {0} features".format(k))

    # if selecaoAtributosAtiva:
    #    train_X_sub, test_X_sub = selecaoAtributo.select_features_RFE(clSVM, x_treino, y_treino, x_teste, k)

    #    print("#####################################")
    #    print("\nClassificador SVM:\n")
    #    tempoExecucao = time.time()
    #    clSVM = svm.SVC(kernel='linear',
    #                   degree=3,
    #                   max_iter=-1)  ## os kernels disponíveis são ('linear', 'poly', 'rbf')
    #   clSVM = clSVM.fit(train_X_sub, y_treino)
    #   y_predito = clSVM.predict(test_X_sub)
    #   tempoExecucao = time.time() - tempoExecucao
    #   print("Tempo de execução:", tempoExecucao, "segundos")

    #    acuracia = accuracy_score(y_predito, y_teste)
    #    print("Acurácia: ", acuracia)
    #    precisao = precision_score(y_predito, y_teste, average='weighted')
    #    print("Precisão: ", precisao)
    #    sensibilidade = recall_score(y_predito, y_teste, average='weighted')
    #    print("Sensibilidade: ", sensibilidade)
    #    erroMedioAbsoluto = mean_absolute_error(y_predito, y_teste)
    #    print("Erro absoluto médio: ", erroMedioAbsoluto)
    #    curvaROC = roc_auc_score(y_predito, y_teste)
    #    print("Curva ROC: ", curvaROC)
    #    f1 = f1_score(y_predito, y_teste, average='weighted')
    #    print("F1: ", f1)
    #    matrizConfusao = confusion_matrix(y_predito, y_teste)
    #    print("Matriz de confusão: \n", matrizConfusao)

if __name__ == '__main__':
    main()