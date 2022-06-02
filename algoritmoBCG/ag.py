import time
import random
import funcoes

def funcaoAptidao(x):
    #Trocar a função manualmente aqui
    # Exemplo com a função esfera -> f(x) = x^2
    resultado = funcoes.rastrigin1dim(x)
    return resultado

# Calcula o valor do fenotipo ou APTIDAO a partir do genotipo
def calculaFenotipo(genotipo):
    fenotipo = funcaoAptidao(genotipo)
    return fenotipo

# Calcula o valor do genotipo a partir do vetorGene
def calculaGenotipo(tamanhoGene,individuo):
    soma = 0
    # Tira -2 porque um é o sinal e o outro é da conversão
    base = 2 ** (tamanhoGene - 2)  # se tamanhoGene = 5, base = 8
    for j in range(1,tamanhoGene):#Começa no 1 pq o 0 é o sinal
        #print("J:", j)
        if individuo[j] == '1':
            soma = soma + base
        base = base / 2
    if individuo[0] == '-':
        soma = -1*soma
    soma = float(soma)
    #print("SOMA:", soma)
    #time.sleep(1)

    return soma

##Declarar o algoritmo - Funciona para declarar o individuo, a população e os operadores
class algoritmoGenetico():

    #Primeiro, definir os parametros iniciais
    #Os limites precisam ficar entre 0 e 2047 -> 2^10
    def __init__(self,limiteInferior,limiteSuperior,tamanhoGene,tamanhoPop, Pc, Pm, maxGeracoes, geracoesEstagnadas):# Adicionar outros?
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior
        self.tamanhoGene = tamanhoGene
        self.tamanhoPop = tamanhoPop
        self.Pc = Pc
        self.Pm = Pm
        self.maxGeracoes = maxGeracoes
        self.geracoesEstagnadas = geracoesEstagnadas

        # calcula o número de bits dos limites no formato binário com sinal
        qtd_bits_x_min = len(bin(limiteInferior).replace('0b', '' if limiteInferior < 0 else '+'))
        qtd_bits_x_max = len(bin(limiteSuperior).replace('0b', '' if limiteSuperior < 0 else '+'))
        #Pode ser obtido assim ou por parâmetro
        self.tamanhoGene = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min
        # gera os individuos da população
        self._iniciaPopulacao()

    #A poppulacao gerada tem tamanho N e cada indivíduo tem vetor gene, genotipo e fenotipo
    def _iniciaPopulacao(self):
        #casasDecimais = int(self.tamanhoGene/2)
        #print("CASAS DECIMAIS:", casasDecimais)
        #time.sleep(1)
        # inicializa uma população de "tam_população" inviduos vazios
        self.populacao = [[] for i in range(self.tamanhoPop)]
        # inicializa uma população auxiliar para a seleção
        self.populacaoAux = [[] for i in range(self.tamanhoPop)]
        #[] é um dataframa sem atributos definidos

        # Preenche a população
        for individuo in self.populacao:
            # para cada individuo da população sorteia números entre "limiteInferior" e "limiteSuperior"
            num = random.randint(self.limiteInferior,self.limiteSuperior)
            #num = random.uniform(self.limiteInferior, self.limiteSuperior)# float
            num_bin = bin(num).replace('0b', '' if num < 0 else '+').zfill(self.tamanhoGene)
            # transforma o número binário resultante em um vetor
            # E passa todas as infos para o vetor
            for bit in num_bin:
                individuo.append(bit)

            # Calcula o genotipo
            genotipo = num #Não precisa calcular pq ele é gerado aleatoriamente inicialmente
            individuo.append(genotipo)
            fenotipo = calculaFenotipo(genotipo)
            individuo.append(fenotipo)
            #print("Individuo:\n", individuo)
            #time.sleep(1)
            #print("Genótipo:", individuo[len(individuo) - 2])
            #print("Aptidão:",individuo[len(individuo) - 1])

    # Encontra o melhor indivíduo e o insere na posição 0 da população
    def encontraEinsereMenor(self):
        i = 0
        menorIndividuo = self.populacao[0]# Só para iniciar
        for individuo in self.populacao:
            if individuo[self.tamanhoGene + 1 ] < menorIndividuo[ self.tamanhoGene + 1 ]:
                #print("tamanho:", self.tamanhoGene)
                #print("\n Ind:",individuo)
                #print("\n Menor:",menorIndividuo)
                menorIndividuo = individuo
            i = i + 1

        self.populacao[0] = menorIndividuo

    #Problemas de minimização
    def encontraMenor(self,x1,x2,x3):
        if (x1[self.tamanhoGene+1] < x2[self.tamanhoGene+1]) & (x1[self.tamanhoGene+1] < x3[self.tamanhoGene+1]):
            return x1
        elif (x2[self.tamanhoGene+1] < x3[self.tamanhoGene+1]) & (x2[self.tamanhoGene+1] < x1[self.tamanhoGene+1]):
            return x2
        else:
            return x3

    def selecao(self):
        """
            Realiza a seleção do individuo mais apto por torneio, considerando N = 3
        """
        #Limpar a aux
        self.populacaoAux.clear()
        # Selecionar 3 indivíduos, até preencher uma nova população
        for j in range(0,self.tamanhoPop):
            individuo_1 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_2 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_3 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuoSelecionado=self.encontraMenor(individuo_1,individuo_2,individuo_3)
            self.populacaoAux.append(individuoSelecionado)
        #print(self.populacaoAux)
        #Agora a população auxiliar precisa ser cruzada

    #Ajusta os indivíduos após o cruzamento, de modo que ele calcula o genótipo e a aptidão
    def _ajustar(self, individuo):
        """
           Caso o individuo esteja fora dos limites de x, ele é ajustado de acordo com o limite superior/infeior
        """
        # Na posição do genótipo, colocamos o genótipo
        individuo[self.tamanhoGene] = calculaGenotipo(self.tamanhoGene, individuo)
        # Na posição do fenótipo, colocamos a aptidão
        individuo[self.tamanhoGene+1] = calculaFenotipo(individuo[self.tamanhoGene])
        #individuo[self.tamanhoGene] corresponde ao genótipo ou X = ponto; f(x) = y
        if individuo[self.tamanhoGene] < self.limiteInferior:
            # se o individuo é menor que o limite mínimo, ele é substituido pelo próprio limite mínimo
            ajuste = bin(self.limiteInferior).replace('0b', '' if self.limiteInferior < 0 else '+').zfill(self.tamanhoGene)
            i = 0
            for bit in ajuste:
                individuo[i] = bit
                i = i + 1
            # Na posição do genótipo, colocamos o genótipo
            individuo[self.tamanhoGene] = calculaGenotipo(self.tamanhoGene, individuo)
            # Na posição do fenótipo, colocamos a aptidão
            individuo[self.tamanhoGene + 1] = calculaFenotipo(individuo[self.tamanhoGene])

        elif individuo[self.tamanhoGene] > self.limiteSuperior:
            # se o individuo é maior que o limite máximo, ele é substituido pelo próprio limite máximo
            ajuste = bin(self.limiteSuperior).replace('0b', '' if self.limiteSuperior < 0 else '+').zfill(self.tamanhoGene)
            i = 0
            for bit in ajuste:
                individuo[i] = bit
                i = i + 1
            # Na posição do genótipo, colocamos o genótipo
            individuo[self.tamanhoGene] = calculaGenotipo(self.tamanhoGene, individuo)
            # Na posição do fenótipo, colocamos a aptidão
            individuo[self.tamanhoGene + 1] = calculaFenotipo(individuo[self.tamanhoGene])
        return individuo

    def cruzamento(self):
        self.populacao.clear()#Limpo para começar a preencher de novo
        i = 0 #Controle de index
        # Se for dentro do range
        if random.randint(1, 100) <= self.Pc: # 70%
            # A cada par de indivíduos, é necessário fazer a troca de genes, logo percorremos os pares até a metade da população
            for individuo in self.populacaoAux:
                # caso o crossover seja aplicado os pais cruzam seus genes e com isso geram dois filhos
                if i%2==0:
                    pai = individuo
                else:
                    mae = individuo

                #Quando for uma geração ímpar, é pq é possível ter filhos e colocar na população original
                if i%2!=0:
                    ponto_de_corte = random.randint(1, self.tamanhoGene - 2)
                    #print("PONTO DE CORTE:", ponto_de_corte)
                    #print("INDEX:", pai[:ponto_de_corte])#Pega a parte esquerda do pai  [***|] do index para trás
                    #print("INDEX:", mae[ponto_de_corte:])#Pega a parte direita da mae  [|****] maior que index pra frente
                    filho_1 = pai[:ponto_de_corte] + mae[ponto_de_corte:]
                    filho_2 = mae[:ponto_de_corte] + pai[ponto_de_corte:]
                    #É preciso ajustar o genótipo e fenótipo
                    #filho_1 = self._ajustar(filho_1)
                    #filho_2 = self._ajustar(filho_2)
                    self.populacao.append(filho_1)
                    self.populacao.append(filho_2)
                i = i + 1 ##Fecha o for
        else:
            for individuo in self.populacaoAux:
                # caso contrário os filhos são cópias exatas dos pais
                self.populacao.append(individuo)

    def mutacao(self):
        """
        Realiza a mutação dos bits de um indiviuo conforme uma dada probabilidade
        (taxa de mutação) e os coloca na população AUXILIAR
        """
        self.populacaoAux.clear()
        for individuo in self.populacao:# cada indivíduo testa a sorte de ser mutado
            if random.randint(1, 100) <= self.Pm:
                quantidade = random.randint(1,int( self.tamanhoGene - 2 ))#Quantidade de bits a serem mutados
                #vetor = random.getrandbits(quantidade)
                #print("VETOR:", vetor)
                for i in range(0,quantidade):
                    if random.randint(1,10) > 5:
                        if individuo[i] == '1':
                            individuo[i] = '0'
                        elif individuo[i] == '0':
                            individuo[i] = '1'
            self.populacaoAux.append(individuo)
            #individuo = self._ajustar(individuo)##Ajustar pra não sair dos limites


    ##FUNÇÃO QUE CALCULA A APTIDAO E FAZ O CONTROLE DA POPULAÇÃO, RECEBE A AUX E COLOCA NA ORIGINAL
    def avaliar(self):
        """
            Avalia as souluções produzidas, associando uma nota/avalição a cada elemento da população
            Lê a população auxiliar e os coloca na população original
        """
        self.populacao.clear()
        #Calcula o genótipo e fenótipo, fazer apenas aqui para evitar calculos desnecessários
        for individuo in self.populacaoAux:
            #self.avaliacao.append(self._funcao_objetivo(individuo))
            individuo = self._ajustar(individuo)#Ajusta os limites e faz os calculos
            self.populacao.append(individuo)

def exibePopulacao(self):##Self é o objeto da classe algoritmoGenetico
    i = 0
    for individuo in self.populacao:
        print("Indivíduo[",i,"]:")
        print(individuo, "\n\t -> Aptidão:",individuo[len(individuo) - 1],"\n")
        i+=1

def exibePopulacaoAux(self):##Self é o objeto da classe algoritmoGenetico
    i = 0
    for individuo in self.populacaoAux:
        print("Indivíduo[", i, "]:")
        print(individuo, "\n\t -> Aptidão:", individuo[len(individuo) - 1], "\n")
        i+=1

def encontraMedia(self):
    media = 0
    for individuo in self.populacao:
        #print("GENOTIPO",individuo[self.tamanhoGene])
        #print("APTIDAO",individuo[self.tamanhoGene+1])
        media = media + individuo[self.tamanhoGene + 1]
        #print("Media",media)
    media = media/len(self.populacao)
    print("media",media)
    print("tamanho",len(self.populacao))
    return media


def exibeMelhorIndividuo(self):
    return self.populacao[0]

def main():
    text_file = open("saidas/saida_ag.txt", "w")#Gravação
    # Gera a população inicial e calcula os valores de aptidão [-1023 a 1023] -> 21 = 1(sinal)+10(inteiro)+10(decimais)
    # (-262143, 262143, 20, 100, 70, 1, 1000, 100)

    limiteInferior = -1000
    limiteSuperior = 1000  # ( 500 TamanhoGene fica em 10 ->+010101001)
    dimensoes = 10
    tamanhoPop = 100
    pc = 70
    pm = 10
    maxGeracoes = 10000
    geracoesEstagnadas = 500
    algoritmo_genetico = algoritmoGenetico(limiteInferior, limiteSuperior, dimensoes, tamanhoPop, pc, pm, maxGeracoes, geracoesEstagnadas)# Já é avaliada depois de criada

    #print("Exibir a população:")
    exibePopulacao(self=algoritmo_genetico)
    algoritmo_genetico.encontraEinsereMenor()#Aplica elistismo na população inicial

    # executa o algoritmo por "num_gerações" ou até estagnar
    estagnacao = 0
    for epoca in range(algoritmo_genetico.maxGeracoes):
        ultimoMelhorIndividuo = algoritmo_genetico.populacao[0]
        # imprime o resultado a cada geração, começando da população original
        #Seleção:
        algoritmo_genetico.selecao()# Seleciona indivíduos e os coloca na população auxiliar
        #exibePopulacao(self=algoritmo_genetico)
        algoritmo_genetico.cruzamento()
        #print("\n#################################\n")
        #print("\n APÓS CRUZAMENTO:")
        #exibePopulacao(self=algoritmo_genetico)
        #exibePopulacaoAux(self=algoritmo_genetico)
        algoritmo_genetico.mutacao()
        #print("\n#################################\n")
        #print("\n APÓS MUTAÇÃO:")
        #exibePopulacaoAux(self=algoritmo_genetico)
        algoritmo_genetico.avaliar()
        #print("\n#################################\n")
        #print("\n APÓS AVALIAÇÃO:")
        #exibePopulacao(self=algoritmo_genetico)
        algoritmo_genetico.encontraEinsereMenor()#Aplica elistismo na população
        mediaGeracao = encontraMedia(self=algoritmo_genetico)
        #print("Geração:", epoca)
        #print("media geração:",mediaGeracao)
        text_file.write("Geração: %d \t %f \t %s \n" % (epoca,mediaGeracao,ultimoMelhorIndividuo[algoritmo_genetico.tamanhoGene+1]))
        ##Verificação de estagnação
        if ultimoMelhorIndividuo == algoritmo_genetico.populacao[0]:
            estagnacao = estagnacao + 1
        else:
            estagnacao = 0
        if estagnacao == algoritmo_genetico.geracoesEstagnadas:
            print("\n\nESTAGNOU!!!!!!!!!")
            break

    print("Número de épocas:",epoca)
    print("Média da última geração:",mediaGeracao)
    print("Melhor indivíduo encontrado:",exibeMelhorIndividuo(self=algoritmo_genetico))
    text_file.close()
    return 0


if __name__ == '__main__':
    main()
