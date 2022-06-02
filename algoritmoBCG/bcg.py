import time
import random
import funcoes
import numpy as np
import math

ponto = [0,0]

def funcaoAptidao(ninho,dimensoes):
    #Trocar a função manualmente aqui
    resultado = funcoes.rastriginCompleta(ninho,dimensoes)
    return resultado

# Calcula o valor do fenotipo ou APTIDAO a partir do genotipo
def calculaAptidao(ninho,dimensoes,posicaoGene):
    #Preciso pegar os valores de genótipo, presente nas posições
    #tamanhoGene*i; i=1
    vetorGenotipo = []
    indice = posicaoGene#Posição onde está o genótipo -> 21
    for i in range(0,dimensoes):
        vetorGenotipo.append(ninho[indice])
        #print(indice)
        indice = indice + posicaoGene + 1 #Atualiza para ir para a próxima -> 21 -> 42
    #print("O que será mandado para a função:", vetorGenotipo)
    aptidao = funcaoAptidao(vetorGenotipo,dimensoes)
    return aptidao

# Calcula o valor do genotipo a partir do vetorGene
#OK
def calculaGenotipo(tamanhoGene,individuo,inicioGene,fimGene):
    soma = 0
    # Tira -2 porque um é o sinal e o outro é da conversão
    base = 2 ** (tamanhoGene - 2)  # se tamanhoGene = 5, base = 8; tamanhoGene = 8, base = 2^6=64
    #print("Dados da calcula genotipo:")
    #print("BASE:",base)
    #time.sleep(100)
    #print("Tamanho Gene:",tamanhoGene)
    #print("Ninho:",individuo)
    #print("Indices:",inicioGene,fimGene)
    #time.sleep(4)
    for j in range(inicioGene+1,fimGene+1):#Começa no 1 pq o 0 é o sinal
        #print("J:", j)
        if individuo[j] == '1':
            soma = soma + base
        base = base / 2
    if individuo[inicioGene] == '-':
        soma = -1*soma
    soma = int(soma)
    #print("SOMAAAA:",soma)
    #time.sleep(3)
    return soma

class buscaCucoGenetico():

    def __init__(self,limiteInferior,limiteSuperior,dimensoes,tamanhoPop,Pa,alfa,beta,pc,pm,maxGeracoes,geracoesEstagnadas):
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior

        self.dimensoes = dimensoes #X, Y ...
        ##self.tamanhoGene = tamanhoGene # Representação dos reais em binário
        self.tamanhoPop = tamanhoPop # Deve ficar entre 20 e 100

        # calcula o número de bits dos limites no formato binário com sinal
        qtd_bits_x_min = len(bin(limiteInferior).replace('0b', '' if limiteInferior < 0 else '+'))
        qtd_bits_x_max = len(bin(limiteSuperior).replace('0b', '' if limiteSuperior < 0 else '+'))

        # Pode ser obtido assim ou por parâmetro
        self.tamanhoGene = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min

        self.Pa = Pa
        self.alfa = alfa # Geralmente a=1
        self.beta = beta # Geralmente 0<=b<=2

        self.pc = pc # Taxa de cruzamento
        self.pm = pm # Taxa de mutação

        self.maxGeracoes = maxGeracoes # De mil até 100k
        self.geracoesEstagnadas = geracoesEstagnadas # Talvez 100 até 10% de MaxGeracoes

        self._iniciaPopulacao()

    def _iniciaPopulacao(self):
        # Inicia a população de ninhos
        self.populacao = [[] for i in range(self.tamanhoPop)]
        # Usado para controlar o AG
        self.populacaoAux = [[] for i in range(self.tamanhoPop)]

        #Vamos considerar que cada ninho tenha exatamente dois ovos -> x e y que compoem f(x,y) = Z
        for ninho in self.populacao:
            #Representa cada ovo
            for dim in range(0,self.dimensoes):
                XN = random.randint(self.limiteInferior, self.limiteSuperior)  # gera um número aleatório
                print("Número aleatório gerado:",XN)

                num_bin = bin(XN).replace('0b', '' if XN < 0 else '+').zfill(self.tamanhoGene) # Converte para binário
                #num_bin = format(struct.unpack('!I', struct.pack('!f', XN))[0], '011b')
                print("num_bin",num_bin)

                for bit in num_bin:
                    ninho.append(bit)
                ninho.append(XN)
            print("Tamanho do gene",self.tamanhoGene)
            print("Ninho sem aptidão calculada:", ninho)
            aptidao = calculaAptidao(ninho,self.dimensoes,self.tamanhoGene)

            ninho.append(aptidao)
            print("Ninho com aptidão calculada:", ninho)
            print("Tamanho do ninho:",len(ninho))
            ultimaPosicao = len(ninho)

            #print("Teste de avaliação/AJUSTE")
            #ninho[0:12] = ['+', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', 1023]
            #ninho[12:24] = ['+', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',1023]
            #print("NINHO ANTES \n", ninho)
            #ninho = self.avaliar(ninho)
            #print("NINHO DEPOIS \n",ninho)
            #time.sleep(100)

        #ORDENAÇÃO
        #A posição dimensoes do ninho (última) sempre é a aptidão
        self.populacao = sorted(self.populacao, key=lambda ninho: ninho[ultimaPosicao-1])#Ordenar de acordo com a qualidade, em ordem crescente
        self.melhorIndividuoDoAG = self.populacao[0]
        print("melhorIndividuoDoAG na inicia_populacao:", self.melhorIndividuoDoAG)

    # Problemas de minimização - x é o individuo
    def encontraMenor(self, x1, x2, x3):
        if (x1[len(x1)-1] < x2[len(x2)-1]) and (x1[len(x1)-1] < x3[len(x3)-1]):
            return x1
        elif (x2[len(x2)-1] < x3[len(x3)-1]) and (x2[len(x2)-1] < x1[len(x1)-1]):
            return x2
        else:
            return x3

    def selecao(self):
        """
            Realiza a seleção do individuo mais apto por torneio, considerando N = 3
        """
        # Limpar a aux
        self.melhorIndividuoDoAG = self.populacao[0]  # Salvar o melhor ninho para garantir o elitismo
        self.populacaoAux.clear()
        # Selecionar 3 indivíduos, até preencher uma nova população
        for j in range(0, self.tamanhoPop):
            individuo_1 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_2 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_3 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            #print("Ninho selecionado 1", individuo_1)
            #print("Ninho selecionado 2", individuo_2)
            #print("Ninho selecionado 3", individuo_3)

            individuoSelecionado = self.encontraMenor(individuo_1, individuo_2, individuo_3)
            self.populacaoAux.append(individuoSelecionado)
        #print("Seleção:",self.populacaoAux)
        #time.sleep(10)
        # Agora a população auxiliar precisa ser cruzada

    def cruzamento(self):
        self.populacao.clear()  # Limpo para começar a preencher de novo
        i = 0  # Controle de index
        # A cada par de indivíduos, é necessário fazer a troca de genes, logo percorremos os pares até a metade da população
        for individuo in self.populacaoAux:
            #print("Indivíduo a ser cruzado:", individuo)
            # caso o crossover seja aplicado os pais cruzam seus genes e com isso geram dois filhos
            if i % 2 == 0:
                pai = individuo
            else:
                mae = individuo

            # Quando for uma geração ímpar, é pq é possível ter filhos e colocar na população original
            if i % 2 != 0:
                # Se for dentro do range
                if random.randint(1, 100) <= self.pc:  # 70%
                    indiceGenotipo = self.tamanhoGene
                    indiceInicioGene = 0
                    indiceFimGene = self.tamanhoGene - 1
                    parteIndividuo_1 = []
                    parteIndividuo_2 = []
                    filho_1 = []
                    filho_2 = []
                    # Para cada dimensão, eu preciso fazer um corte nos genes e recombinar um par de indivíduos
                    for dimensao in range(0, self.dimensoes):
                        ponto_de_corte = random.randint(indiceInicioGene + 1, indiceFimGene - 1)  # -2

                        #print("PONTO DE CORTE:", ponto_de_corte)
                        #print("INDEX PAI:", pai[indiceInicioGene:ponto_de_corte])  # Pega a parte esquerda do pai  [***|] do index para trás
                        #print("INDEX MAE:", mae[ponto_de_corte:indiceFimGene + 2])  # Pega a parte direita da mae  [|****] maior que index pra frente

                        parteIndividuo_1.append(pai[indiceInicioGene:ponto_de_corte] + mae[ponto_de_corte:indiceFimGene + 2])
                        parteIndividuo_2.append(mae[indiceInicioGene:ponto_de_corte] + pai[ponto_de_corte:indiceFimGene + 2])

                        indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                        indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                        indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza os indices

                    # Converte os filhos em array único e para lista para ficar igual antes
                    filho_1 = np.asarray(parteIndividuo_1).flatten().tolist()
                    filho_2 = np.asarray(parteIndividuo_2).flatten().tolist()

                    print("FILHO A", filho_1)
                    print("FILHO B", filho_2)

                    self.populacao.append(filho_1)
                    self.populacao.append(filho_2)
                else:
                    # caso contrário os filhos são cópias exatas dos pais
                    # É preciso dar um pop para evitar problemas na próxima geração
                    if (len(pai)) > ((self.dimensoes*self.tamanhoGene)+self.dimensoes):
                        print("PAI:",pai)
                        print("Tamanho pai:",len(pai))
                        #time.sleep(3)
                        while (len(pai) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes )):
                            pai.pop()
                    if (len(mae)) > ((self.dimensoes * self.tamanhoGene) + self.dimensoes):
                        print("MAE:", mae)
                        print("Tamanho mae:", len(mae))
                        while (len(mae) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes)):
                            mae.pop()

                    self.populacao.append(pai)
                    self.populacao.append(mae)

            i = i + 1  ##Fecha o for, controle de par e ímpar

    def mutacao(self):
        """
            Realiza a mutação dos bits de um indiviuo conforme uma dada probabilidade
            (taxa de mutação pm) e os coloca na população AUXILIAR
        """
        self.populacaoAux.clear()
        for individuo in self.populacao:  # cada indivíduo testa a sorte de ser mutado
            if random.randint(1, 100) <= self.pm:
                # quantidade = random.randint(1, int(self.tamanhoGene/2))  # Quantidade de bits a serem mutados
                # vetor = random.getrandbits(quantidade)
                # print("VETOR:", vetor)
                indiceGenotipo = self.tamanhoGene
                indiceInicioGene = 0
                indiceFimGene = self.tamanhoGene - 1
                for dimensoes in range(0,self.dimensoes):
                    for i in range(indiceInicioGene+1, indiceFimGene+1):#Mais 1 para evitar mudar o sinal
                        if random.randint(1, 10) >6: # Com isto, até 40% dos genes podem sofrer mutação -> x<=40%
                            if individuo[i] == '1':
                                individuo[i] = '0'
                            elif individuo[i] == '0':
                                individuo[i] = '1'
                            elif individuo[i] == '+':
                                individuo[i] = '-'
                            elif individuo[i] == '-':
                                individuo[i] = '+'
                    indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                    indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                    indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1
            print("Indivíduo mutado:",individuo)
            #time.sleep(1)
            while (len(individuo) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes)):
                individuo.pop()
                print("Erro da mutação")
                time.sleep(100)
            self.populacaoAux.append(individuo)
        #i = 0
        #for ninho in self.populacaoAux:
        #    print("Indivíduo[", i, "]:")
        #    print(ninho, "\n\t -> Tamanho:", len(ninho), "\n")
        #    i += 1
        #time.sleep(1)


    ##FUNÇÃO QUE CALCULA A APTIDAO E FAZ O CONTROLE DA POPULAÇÃO, RECEBE A PopAUX E COLOCA NA ORIGINAL
    #No caso do AG
    def avaliarAG(self):
        """
            Avalia as souluções produzidas, associando uma nota/avalição a cada elemento da população
            Lê a população auxiliar e os coloca na população original
        """
        self.populacao.clear()#Limpa inicialmente
        # Calcula o genótipo e fenótipo, fazer apenas aqui para evitar calculos desnecessários

        for ninho in self.populacaoAux:
            #print("NINHO_AUX",ninho)
            #print("TAMANHO_AUX",len(ninho))
            if len(ninho) > ( (self.tamanhoGene*self.dimensoes) + self.dimensoes ):
                time.sleep(10)
                print("Algo de errado!")
                print("NINHO_AUX", ninho)
                print("TAMANHO_AUX", len(ninho))
                while ( len(ninho) > ( (self.tamanhoGene*self.dimensoes) + self.dimensoes ) ):
                    ninho.pop()
                print("NINHO_AUX", ninho)
                print("TAMANHO_AUX", len(ninho))
                time.sleep(10)

        for ninho in self.populacaoAux:

            # inicialização dos indices
            indiceGenotipo = self.tamanhoGene
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1

            for dimensoes in range(0, self.dimensoes):
                # Na posição do genótipo, colocamos o genótipo
                ninho[indiceGenotipo] = int(calculaGenotipo(self.tamanhoGene, ninho, indiceInicioGene, indiceFimGene))

                # ninho[indiceGenotipo] = int(ninho[indiceGenotipo])
                if ninho[indiceGenotipo] > self.limiteSuperior:
                    #ninho[indiceGenotipo] = self.limiteSuperior
                    numero_bin = bin(self.limiteSuperior).replace('0b', '' if self.limiteSuperior < 0 else '+').zfill(self.tamanhoGene)  # Converte para binário
                    i = 0
                    for k in range(indiceInicioGene,indiceFimGene+1):
                        #print("numero_bin[i]:", numero_bin[i])
                        ninho[k] = numero_bin[i]
                        i = i + 1
                    #for bit in numero_bin:
                    #    ninho.append(bit)
                    #    print("bit:",bit)
                    ninho[indiceGenotipo] = self.limiteSuperior

                elif ninho[indiceGenotipo] < self.limiteInferior:
                    #ninho[indiceGenotipo] = self.limiteInferior
                    numero_bin = bin(self.limiteInferior).replace('0b', '' if self.limiteInferior < 0 else '+').zfill(self.tamanhoGene)  # Converte para binário
                    i = 0
                    for k in range(indiceInicioGene, indiceFimGene + 1):
                        #print("numero_bin[i]:", numero_bin[i])
                        ninho[k] = numero_bin[i]
                        i = i + 1
                    ninho[indiceGenotipo] = self.limiteSuperior

                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1

            # print("Ninho após genotipo:",ninho)
            # time.sleep(4)

            # Ultima posicao guarda a aptidão
            aptidao = calculaAptidao(ninho, self.dimensoes, self.tamanhoGene)
            ninho.append(aptidao)

            # Garante que o ninho tenha o tamanho certo
            if (len(ninho) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes + 1)):
                print("ninho wrong:", ninho)
                print("tamanho:", len(ninho))
                #time.sleep(1)
                while (len(ninho) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes + 1)):
                    ninho.pop()
            print("Ninho após ajuste completo\n:", ninho)
            # time.sleep(1)

            self.populacao.append(ninho)

        self.populacao[self.tamanhoPop-1] = self.melhorIndividuoDoAG  # Salvar o melhor ninho para garantir o elitismo, troca o indivíduo 0
        print("melhorIndividuoDoAG na Avaliar AG:", self.melhorIndividuoDoAG)

    #OK
    def avaliar(self,ninho):
        """
            Avalia a solução produzida, associando uma aptidão ao ninho
            e faz o controle dos limites passados
        """
        ##Ajusta os valores de x e y para ficarem dentro dos limites
        #for dimensao in range(0,self.dimensoes):
        #    if ninho[len(ninho)-1] < self.limiteInferior:
        #        ninho[len(ninho)-1] = self.limiteInferior
        #    elif ninho[len(ninho)-1] > self.limiteSuperior:
        #        ninho[len(ninho)-1] = self.limiteSuperior
        #Ajustar o valor de cada genótipo, referente a cada dimensão para ficar
        #dentro dos limites

        indiceGenotipo = self.tamanhoGene ## Posição onde está o primeiro genótipo -> 21
        indiceInicioGene = 0
        indiceFimGene = self.tamanhoGene - 1
        num_bin = []  # Limpa o vetor para o próximo genótipo
        for dimensao in range(0,self.dimensoes):
            if ninho[indiceGenotipo] < self.limiteInferior:
                #preciso atualizar a linha toda!
                ninho[indiceGenotipo] = self.limiteInferior
                num_bin = bin(self.limiteInferior).replace('0b', '' if self.limiteInferior < 0 else '+').zfill(self.tamanhoGene)  # Converte para binário
                i=0
                for j in range(indiceInicioGene,indiceFimGene+1):
                    ninho[j] = num_bin[i]
                    i+=i
                num_bin = [] # Limpa o vetor para o próximo genótipo
            elif ninho[indiceGenotipo] > self.limiteSuperior:
                #preciso atualizar a linha toda!
                ninho[indiceGenotipo] = self.limiteSuperior
                num_bin = bin(self.limiteSuperior).replace('0b', '' if self.limiteSuperior < 0 else '+').zfill(self.tamanhoGene)  # Converte para binário
                i=0
                for j in range(indiceInicioGene,indiceFimGene+1):
                    ninho[j] = num_bin[i]
                    i+=i
                num_bin = [] # Limpa o vetor para o próximo genótipo
            #Atualizar os indices
            indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
            indiceFimGene = indiceFimGene + self.tamanhoGene + 1
            indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42
        #Ultima posicao guarda a aptidão
        aptidao = calculaAptidao(ninho, self.dimensoes, self.tamanhoGene)
        ninho[len(ninho)-1] = aptidao
        #ninho.append(aptidao)
        return ninho

    def vooLevy(self,beta,Xmelhor,Xi):
        # Para calcular essa distribuição, precisamos de várias variáveis:
        # U; V ; BETA; SigmaU e SigmaV = 1

        ## A principio certo SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        #print("Sigma Antes:",SigmaU)
        #Verificar se gamma é custoso de ser calculado
        #Adicionar o salto de forma mais rara
        SigmaU = np.power(SigmaU, (1 / beta)) * 1000000  # Ajuste
        U = random.gauss(0,SigmaU)
        V = random.gauss(0,1)
        L = (U / (np.power(np.fabs(V), 1 / beta))) * (Xmelhor - Xi)
        #print("SIGMA",SigmaU)
        #print("U",U)
        #print("V",V)
        #print("L(Beta):",L)
        #time.sleep(1)
        return L

    #Função que gera a pertubação causada pelo Cuco, a partir da melhor posição
    def geraCucoPorLevy(self,ninho):
        #Primeiro, temos duas buscas, local e global.
        # A local é:
        # X(t+1) = X(t) + alfa(1) * s[entre 0 e 1] * H(Pa - E) * (Xp - Xq)
        #indiceP = random.randint(1,self.tamanhoPop-1)
        #indiceQ = random.randint(1, self.tamanhoPop - 1)
        # X(t+1) = X(t) + 1*random.uniform(0,1)*np.heaviside(self.Pa - random.uniform(0,1),0.5) * (self.populacao[indiceP][dim] - self.populacao[indiceQ][dim])
        # A global é:
        #Vamos focar na global, já que a outra tem elementos de DE
        # X(t+1) = X(t) + alfa*L(beta), aqui alfa deve ser a escala do salto

        novoNinho = [[] for i in range(0, (self.dimensoes*self.tamanhoGene)+(self.dimensoes)+1 )]# um gene cada dimensão um espaço para cada genotipo mais aptidao
        #Atualiza cada ovo ou soluções para compor a nova

        #Vamos adicionar variação, a busca local deveria ser mais frequente
        #logo:
        if random.randint(1,100) < 50:#Busca global
            #Indices iniciais, é preciso calcular os genes de cada dimensão
            indiceGenotipo = self.tamanhoGene  ## Posição onde está o primeiro genótipo -> 21
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1
            for i in range(0,self.dimensoes):
                #vamos ter que trocar os indices, e calcular o genótipo em binário
                L = self.vooLevy(self.beta,self.populacao[0].__getitem__(indiceGenotipo),ninho[indiceGenotipo])
                print("GLOBAL:", L)
                #Se passar dos limites, coloque dentro de novo
                if int(ninho[indiceGenotipo] + self.alfa*L) >= self.limiteSuperior:
                    novoNinho[indiceGenotipo] = self.limiteSuperior
                elif int(ninho[indiceGenotipo] + self.alfa*L) <= self.limiteInferior:
                    novoNinho[indiceGenotipo] = self.limiteInferior
                else:
                    novoNinho[indiceGenotipo] = int(ninho[indiceGenotipo] + self.alfa*L)#Converte para inteiro

                num_bin=bin(novoNinho[indiceGenotipo]).replace('0b', '' if novoNinho[indiceGenotipo] < 0 else '+').zfill(self.tamanhoGene)#Converte para binário
                print("num_bin:",num_bin)
                #time.sleep(5)
                #print("NUM_BIN:",num_bin)
                #print("Genótipo aqui:",novoNinho[indiceGenotipo])
                k = 0
                for j in range(indiceInicioGene, indiceFimGene + 1):
                    novoNinho[j] = num_bin[k]
                    k = k + 1
                num_bin = []  # Limpa o vetor para o próximo genótipo
                #ponto[i] = ponto[i] + L

                #print("indices:")
                #print("indiceInicioGene", indiceInicioGene)
                #rint("indiceFimGene", indiceFimGene)
                #print("indiceGenotipo", indiceGenotipo)
                #print("NOVO ninho:",novoNinho)

                # Atualizar os indices
                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42

        else:#Busca local

            # Indices iniciais, é preciso calcular os genes de cada dimensão
            indiceGenotipo = self.tamanhoGene  ## Posição onde está o primeiro genótipo -> 21
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1

            indiceP = random.randint(1, self.tamanhoPop - 1)
            indiceQ = random.randint(1, self.tamanhoPop - 1)
            while indiceP==indiceQ:     #Pra serem soluções diferentes
                indiceQ = random.randint(1, self.tamanhoPop - 1)
            for i in range(0, self.dimensoes):

                # 1*random.uniform(0,1)*np.heaviside((self.Pa)/100 - random.uniform(0,1),0.5)*(self.populacao[indiceP][i]-self.populacao[indiceQ][i])
                treco = 1 * random.uniform(0, 1) * np.heaviside((50) / 100 - random.uniform(0, 1), 0.5) * (self.populacao[indiceP][indiceGenotipo] - self.populacao[indiceQ][indiceGenotipo])
                print("LOCAL:", treco)

                if int(ninho[indiceGenotipo] + treco) >= self.limiteSuperior:
                    novoNinho[indiceGenotipo] = self.limiteSuperior
                elif int(ninho[indiceGenotipo] + treco) <= self.limiteInferior:
                    novoNinho[indiceGenotipo] = self.limiteInferior
                else:
                    novoNinho[indiceGenotipo] = int(ninho[indiceGenotipo] + treco)  # Converte para inteiro

                num_bin = bin(novoNinho[indiceGenotipo]).replace('0b','' if novoNinho[indiceGenotipo] < 0 else '+').zfill(self.tamanhoGene)  # Converte para binário
                k = 0
                for j in range(indiceInicioGene, indiceFimGene + 1):
                    novoNinho[j] = num_bin[k]
                    k = k + 1
                num_bin = []  # Limpa o vetor para o próximo genótipo

                # Atualizar os indices
                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42

            #print("LOCAL:", ponto)


        #Faz o ajuste do novo ninho e calcula sua aptidão
        novoNinho = self.avaliar(novoNinho)
        #print("NOVO ninho:", novoNinho)
        #time.sleep(2)
        return novoNinho

    def abandonaPa(self):
        complementoPb = 100 - self.Pa  # Pb
        indicePa = (int)((self.tamanhoPop * complementoPb) / 100) - 1  # Indice onde começam as piores soluções

        for k in range(indicePa, self.tamanhoPop):
            #print("Olhar o K:",k)
            #print("Ninho Antigo",self.populacao[k])
            self.populacao[k] = self.geraNinhoAleatorio(self.populacao[k])
            #print("Ninho Novo", self.populacao[k])
            #time.sleep(10)


    def geraNinhoAleatorio(self, ninhoAleatorio):
        #ninhoAleatorio = [[] for i in range(self.dimensoes+1)]
        #print("Ninho enviado", ninhoAleatorio)
        ninhoAleatorio.clear() # preciso limpar ele para que ele receba o novo ninho
        for dim in range(0,self.dimensoes):
            XN = random.randint(self.limiteInferior, self.limiteSuperior)  # gera um número aleatório
            #print("Número aleatório gerado:",XN)
            num_bin = bin(XN).replace('0b', '' if XN < 0 else '+').zfill(self.tamanhoGene) # Converte para binário
            #print("num_bin",num_bin)
            for bit in num_bin:
                ninhoAleatorio.append(bit)
            ninhoAleatorio.append(XN)
        #print("Tamanho do gene",self.tamanhoGene)

        aptidao = calculaAptidao(ninhoAleatorio,self.dimensoes,self.tamanhoGene)
        ninhoAleatorio.append(aptidao)

        ninhoAleatorio = self.avaliar(ninhoAleatorio)
        #print("GERA ninho aleatorio",ninhoAleatorio)
        #time.sleep(100)
        return ninhoAleatorio

#Ok
def encontraMedia(self):
    media = 0
    for individuo in self.populacao:
        media = media + individuo[len(individuo)-1]
        #print("Media",media)
    media = media/len(self.populacao)
    #print("media",media)
    #print("tamanho",len(self.populacao))
    return media

#OK
def exibePopulacao(self):
    i = 0
    for ninho in self.populacao:
        print("Indivíduo[", i, "]:")
        print(ninho, "\n\t -> Aptidão:", ninho[len(ninho)-1], "\n")
        i += 1

#OK
def exibeMelhorIndividuo(self):
    ninhoBom = self.populacao[0]
    print("Melhor aptidão encontrada:", ninhoBom[len(ninhoBom)-1])
    return self.populacao[0]

def main():
    novoNinho_file = open("saidas/saida_bcg.txt", "w")  # Gravação
    melhorNinho_file = open("saidas/melhor_bcg.txt", "w")  # Gravação
    medias_file = open("saidas/media_bcg.txt", "w")  # Gravação

    ativaAG = 1 #Controla se o AG vai ser executado também

    limiteInferior = -1000
    limiteSuperior = 1000 #( 500 TamanhoGene fica em 10 ->+010101001)
    dimensoes = 10
    tamanhoPop = 100
    Pa = 10
    alfa = 1
    beta = 2
    pc = 70
    pm = 10
    maxGeracoes = 10000
    geracoesEstagnadas = 500
    geracaoEvolucao = 10

    busca_Cuco_Genetico = buscaCucoGenetico(limiteInferior,limiteSuperior,dimensoes,tamanhoPop,Pa,alfa,beta,pc,pm,maxGeracoes,geracoesEstagnadas)
    #Todos os ninhos são avaliados inicialmente
    exibePopulacao(self=busca_Cuco_Genetico)

    estagnacao = 0
    PB = int((busca_Cuco_Genetico.tamanhoPop * (100 - busca_Cuco_Genetico.Pa)) / 100)  # 1 - Pa
    tempoExecucao = time.time()
    for epoca in range(busca_Cuco_Genetico.maxGeracoes):
        ultimoMelhorIndividuo = busca_Cuco_Genetico.populacao[0]  # As soluções ficam em ordem crescente, a melhor fica na posição 0
        ultimaMelhorAptidao = ultimoMelhorIndividuo[len(ultimoMelhorIndividuo)-1]
        #exibePopulacao(self=busca_Cuco)

        #Gerar cuco por voo de Levy -> usar um dos melhores/pior indivíduo como semente
        indiceBom = random.randint(1,PB) # Meu I
        #print("Escolhi um bom individuo i=",indiceBom)
        #print("\n->",busca_Cuco.populacao[indiceBom])
        novoNinho = busca_Cuco_Genetico.geraCucoPorLevy(busca_Cuco_Genetico.populacao[indiceBom])
        #time.sleep(1)

        #Salvar o novo ninho descoberto a cada geração
        novoNinho_file.write("%s \n" % (novoNinho) )

        #Escolher o ninho J aleatório
        indiceJ = random.randint(0, busca_Cuco_Genetico.tamanhoPop - 1)
        ninhoJ = busca_Cuco_Genetico.populacao[indiceJ]
        #print("J:",indiceJ,"\nNINHO J",ninhoJ)
        print("NOVO CUCO:", novoNinho)
        print("NINHO J:", ninhoJ)
        #time.sleep(1)


        #Se novoCuco.apt < ninhoJ.apt -> troque, se trocou, ordene! é O(1)
        if novoNinho[len(novoNinho)-1] < ninhoJ[len(novoNinho)-1]:
            #Associa o novo ninho ao ninho J
            for i in range(0,busca_Cuco_Genetico.dimensoes+1):
                busca_Cuco_Genetico.populacao[indiceJ][i] = novoNinho[i]

            busca_Cuco_Genetico.populacao = sorted(busca_Cuco_Genetico.populacao,key=lambda ninho: ninho[len(ninho)-1])  # Ordenar de acordo com a qualidade, ordem crescente
            novoNinho.clear()#limpar para a proxima epoca
            #print("\n#######################################\n")

        #Parte do algoritmo genético: evolução
        #Executa a cada x gerações - as operações são pesadas
        if ((epoca+1)%geracaoEvolucao == 0) and (epoca>8) and (ativaAG==1):
            #Seleção
            busca_Cuco_Genetico.selecao()
            #Cruzamento
            busca_Cuco_Genetico.cruzamento()
            #Mutação
            busca_Cuco_Genetico.mutacao()
            #for indiv in busca_Cuco_Genetico.populacao:
            busca_Cuco_Genetico.avaliarAG()
            #O elistismo seria aplicado aqui, porém a BC já o faz naturalmente
            busca_Cuco_Genetico.populacao = sorted(busca_Cuco_Genetico.populacao, key=lambda ninho: ninho[len(ninho) - 1]) # Ordenar de acordo com a qualidade
        else:
            # Abandone a porção Pa -> Gere novas soluções aleatórias aqui
            busca_Cuco_Genetico.abandonaPa()
            # Ordene de novo de acordo com a aptidão e salve o melhor
            busca_Cuco_Genetico.populacao = sorted(busca_Cuco_Genetico.populacao,key=lambda ninho: ninho[len(ninho)-1])  # Ordenar de acordo com a qualidade
            #exibePopulacao(self=busca_Cuco_Genetico)
            #time.sleep(1)

        mediaGeracao = encontraMedia(self=busca_Cuco_Genetico)#Pega a média de aptidão da população

        medias_file.write("Geracao: %d \t %f \t %s \n" % (epoca, mediaGeracao, ultimoMelhorIndividuo[len(ultimoMelhorIndividuo)-1]))
        melhorNinho_file.write( "%s\n" % (ultimoMelhorIndividuo) )

        testeAqui = busca_Cuco_Genetico.populacao[0]
        aqui = testeAqui[len(testeAqui)-1]
        ##Verificação de estagnação
        #if ultimaMelhorAptidao == busca_Cuco_Genetico.populacao[0]:
        if ultimaMelhorAptidao == aqui:
            estagnacao = estagnacao + 1
        else:
            estagnacao = 0
        if estagnacao == busca_Cuco_Genetico.geracoesEstagnadas:
            print("\n\n## -- ESTAGNOU! -- ##")
            break


        exibePopulacao(self=busca_Cuco_Genetico)
        print("##ÉPOCA:", epoca)
        #time.sleep(1)

    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:",tempoExecucao)
    print("Número de épocas:", epoca)
    print("Média da última geração:", mediaGeracao)
    print("Melhor ninho encontrado:", exibeMelhorIndividuo(self=busca_Cuco_Genetico))

    novoNinho_file.close()
    melhorNinho_file.close()
    medias_file.close()
    return 0

if __name__ == '__main__':
    main()