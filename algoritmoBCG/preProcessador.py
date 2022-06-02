import re
import string
import palavrasParada
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from unicodedata import normalize
import time
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import StemmerI
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer

from nltk.stem import WordNetLemmatizer
lematizador = WordNetLemmatizer()

# init stemmer
#radicalizador = PorterStemmer()
radicalizador = SnowballStemmer(language='english')
#radicalizador = LancasterStemmer()


def meu_preprocessador(frase): ##Text é uma frase
    #print("Texto entrada:",text)

    frase = frase.lower()
    frase = frase.replace("rt", "")


    frase = normalize('NFKD', frase).encode('ASCII', 'ignore').decode('ASCII')  # Remove pontuação, o que não está na tabela ASCII
    frase = re.sub('\?|\.|\!|\/|\;|\:', '', frase) #Exemplo de remoção por expressão regular
    frase = re.sub("\\W", " ", frase)  # remove special chars, caracter não alfa-numérico
    frase = re.sub("\\s+(in|the|all|for|and|on)\\s+", " ", frase)  # normalize certain words _connector_
    frase = re.sub("http\S+", "", frase)
    frase = re.sub("https\S+", "", frase)
    frase = re.sub("@\S+", "", frase) #Remove nomes de usuário
    frase = frase.replace("http", "")
    frase = frase.replace("'", "")

    ##Emojis 'padrão'
    frase = frase.replace("=D","happy")
    frase = frase.replace("=)", "happy")
    frase = frase.replace(":)", "happy")
    frase = frase.replace(":D", "happy")
    frase = frase.replace(":‑)", "happy")
    frase = frase.replace(":-]", "happy")
    frase = frase.replace(":]", "happy")
    frase = frase.replace(":-3", "happy")
    frase = frase.replace(":3", "happy")



    # radicalização
    words = re.split("\\s+", frase)  # It simply means: slice the input string s on the given regular expression.
    palavras_radicalizadas = [radicalizador.stem(word=word)for word in words]
    frase = ' '.join(palavras_radicalizadas)




    # Lemas
    words = re.split("\\s+", frase)  # It simply means: slice the input string s on the given regular expression.
    palavras_lemas = [lematizador.lemmatize(word=word) for word in words]
    frase = ' '.join(palavras_lemas)


    padraoBuscado = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
    #print("Texto saída:", frase)
    #time.sleep(1)
    return padraoBuscado.sub(r'', frase)

def preprocessadorParaSalvar(frase, contexto, ativaPOStag):  ##Text é uma frase
    # print("Texto entrada:",text)
    frase = frase.lower() ##para facilitar o trabalho do contexto
    frase = frase.replace("rt","") # Remove marcação de retweet
    frase = re.sub("http\S+", "url", frase)  # Remove links
    frase = re.sub("https\S+", "url", frase)  # Remove links
    frase = re.sub('@[^\s]+', '', frase)  # Remove nome de usuário
    frase = re.sub('\?|\.|\!|\/', '', frase)  # Exemplo de remoção por expressão regular, remove pontuação

    if contexto == "tech":
        frase = frase.replace("apple", "company")
        frase = frase.replace("microsoft", "company")
        frase = frase.replace("google", "company")
        frase = frase.replace("android", "system")
        frase = frase.replace("4s", "model")
        frase = frase.replace("siri", "system")
        frase = frase.replace("ice cream", "system")
        frase = frase.replace("ice cream sandwich", "system")
        frase = frase.replace("ics", "system")
        frase = frase.replace("ios", "system")
        frase = frase.replace("ios5", "system")
        frase = frase.replace("samsung", "company")
        frase = frase.replace("galaxynexus", "model")
        frase = frase.replace("galaxy nexus", "model")
        frase = frase.replace("nexus", "model")
        frase = frase.replace("galaxy", "model")
        frase = frase.replace("twitter", "company")
        frase = frase.replace("windows", "company")
        frase = frase.replace("stevejobs", "businessperson")
        frase = frase.replace("steve", "businessperson")
        frase = frase.replace("jobs", "businessperson")
        frase = frase.replace("ipod", "model")
        frase = frase.replace("ipad", "model")
        frase = frase.replace("iphone", "model")
        frase = frase.replace("iphone4s", "model")
        frase = frase.replace("icloud", "system")
        frase = frase.replace("imessage", "system")
        frase = frase.replace(" u ", "you")
        frase = frase.replace(":)", "happy")
        frase = frase.replace("itunes", "system")
        frase = frase.replace("eric holder", "person")
        frase = frase.replace("amazon", "system")
        frase = frase.replace("macbook", "model")
        frase = frase.replace("youtube", "media")
        print("pós thesauros:", frase)

        #Remoção de stop words por contexto
        stop_words_tech = set(palavrasParada.minhas_paralavras_de_parada)#Conjunto de stop words
        frase_tokenizada = word_tokenize(frase)#tokeniza a frase
        frase_filtrada = [w for w in frase_tokenizada if not w in stop_words_tech]
        frase = ' '.join(frase_filtrada)

        #print("Deu tech")
    elif contexto == "airline":
        frase = frase.replace("southwestair", "company")
        frase = frase.replace("united", "company")
        frase = frase.replace("jetblue", "company")
        frase = frase.replace("usairways", "company")
        frase = frase.replace("virginamerica", "company")
        frase = frase.replace("&amp", "")
        frase = frase.replace("dm", "message")
        frase = frase.replace(" u ", "you")
        frase = frase.replace("rebook", "book")
        frase = frase.replace("love_dragonss", "user")
        frase = frase.replace("jfk", "airplane")
        frase = frase.replace("youtube", "media")
        print("pós thesauros:", frase)

        stop_words_tech = set(palavrasParada.minhas_paralavras_de_parada)  # Conjunto de stop words
        frase_tokenizada = word_tokenize(frase)  # tokeniza a frase
        frase_filtrada = [w for w in frase_tokenizada if not w in stop_words_tech]
        frase = ' '.join(frase_filtrada)

        #print("Deu airline")
    elif contexto == "politics":
        #print("Deu politics")
        frase = frase.replace("g1", "journal")
        frase = frase.replace("bolsonaro", "president")
        frase = frase.replace("bolsonaro17", "president")
        frase = frase.replace("jair", "president")
        frase = frase.replace("haddad_fernando", "haddad")
        frase = frase.replace("eleições2018", "elections2018")
        frase = frase.replace("#folha", "journal")
        frase = frase.replace("datafolha", "journal")
        frase = frase.replace("guia_folha", "journal")
        frase = frase.replace("são paulo", "sp")
        frase = frase.replace("estadaopolitica", "journal")
        frase = frase.replace("g1sp", "journal")
        frase = frase.replace("youtube", "journal")
        frase = frase.replace("via", "by")
        frase = frase.replace("paulo guedes", "ministry")
        frase = frase.replace("netflix", "company")
        frase = frase.replace(":)", "happy")
        frase = frase.replace(":(", "sad")
        frase = frase.replace("oglobo_rio", "journal")
        frase = frase.replace("estadaointer", "journal")
        frase = frase.replace("hj ", "today")
        frase = frase.replace("youtube", "media")
        frase = frase.replace("mt", "much")
        frase = frase.replace("bolsa","exchange")
        print("pós thesauros:", frase)

        stop_words_tech = set(palavrasParada.minhas_paralavras_de_parada)  # Conjunto de stop words
        frase_tokenizada = word_tokenize(frase)  # tokeniza a frase
        frase_filtrada = [w for w in frase_tokenizada if not w in stop_words_tech]
        frase = ' '.join(frase_filtrada)

    else:
        print("PARE!")
        time.sleep(100000)

    print("pós stop-words:",frase)

    ##Emojis 'padrão'
    frase = frase.replace("=)", "happy")
    frase = frase.replace(":)", "happy")
    frase = frase.replace(":‑)", "happy")
    frase = frase.replace(":-]", "happy")
    frase = frase.replace(":]", "happy")
    frase = frase.replace(":-3", "happy")
    frase = frase.replace(":3", "happy")
    frase = frase.replace("=(", "sad")
    frase = frase.replace(":(", "sad")
    frase = frase.replace(":‑(", "sad")
    frase = frase.replace(":-[", "sad")
    frase = frase.replace(":[", "sad")
    frase = frase.replace(":/", "sad")
    frase = frase.replace(":\\", "sad")

    #Última etapa
    frase = normalize('NFKD', frase).encode('ASCII', 'ignore').decode('ASCII') #Remove pontuação, o que não está na tabela ASCII
    frase = re.sub('\?|\.|\!|\/|\;|\:', '', frase)  # Exemplo de remoção por expressão regular, remove pontuação
    frase = re.sub("\\W", " ", frase)  # remove special chars, caracter não alfa-numérico
    frase = re.sub("\\s+(in|the|all|for|and|on)\\s+", " ", frase)  # normalize certain words _connector_
    frase = frase.replace("'", "")

    print("pós limpeza final:", frase)

    # radicalização
    # for w in words:
    #    print(w, " : ", radicalizador.stem(w))
    words = re.split("\\s+", frase)  # Isso tokeniza as palavras da frase, após uma breve limpeza
    palavras_radicalizadas = [radicalizador.stem(word=word) for word in words]
    frase = ' '.join(palavras_radicalizadas)
    print("após radicalização:", frase)

    # Lemas
    # lemas = re.split("\\s+",frase)
    # frase = "studies studying cries cry"
    # tokenization = nltk.word_tokenize(frase)
    #for w in frase:
    #    print("Lemma for {} is {}".format(w, lematizador.lemmatize(w)))
    #    palavras_lemas = lematizador.lemmatize(w)
    words = re.split("\\s+", frase)  # It simply means: slice the input string s on the given regular expression.
    palavras_lemas = [lematizador.lemmatize(word=word) for word in words]
    frase = ' '.join(palavras_lemas)
    print("após lemarização:", frase)

    #Remoção de sujeira final
    padraoBuscado = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)

    # Vamos ver com o POS_TAG
    if ativaPOStag:
        frase = frase.replace("url", "")  # Remove esta palavra que atrapalha o POS
        frase_tokenizada = word_tokenize(frase)  # tokeniza a frase
        frase_tokenizada_e_pos = pos_tag(frase_tokenizada)  # cria os POS_tag em cada token
        print(frase_tokenizada_e_pos)
        frase = ''
        for algo in frase_tokenizada_e_pos:  # algo é um token e uma tag
            # print(frase_tokenizada_e_pos[linha][0])
            palavraCompleta = str(algo[0]) + '_' + str(algo[1])
            # algo=frase_tokenizada_e_pos[linha][0]+frase_tokenizada_e_pos[linha][1]
            print(algo)
            print(palavraCompleta)
            frase = frase + ' ' + palavraCompleta

        print("pós POS_TAG:", frase)
        # time.sleep(100)
    # Fim POS_TAG

    print("Texto saída:", frase)


    # time.sleep(1)
    return padraoBuscado.sub(r'', frase)

'''
Abbreviation	Meaning
CC	            coordinating conjunction
CD	            cardinal digit
DT	            determiner
EX	            existential there
FW	            foreign word
IN	            preposition/subordinating conjunction
JJ	            This NLTK POS Tag is an adjective (large)
JJR	            adjective, comparative (larger)
JJS	            adjective, superlative (largest)
LS	            list market
MD	            modal (could, will)
NN	            noun, singular (cat, tree)
NNS	            noun plural (desks)
NNP	            proper noun, singular (sarah)
NNPS	        proper noun, plural (indians or americans)
PDT	            predeterminer (all, both, half)
POS	            possessive ending (parent\ 's)
PRP	            personal pronoun (hers, herself, him,himself)
PRP$	        possessive pronoun (her, his, mine, my, our )
RB	            adverb (occasionally, swiftly)
RBR	            adverb, comparative (greater)
RBS	            adverb, superlative (biggest)
RP	            particle (about)
TO	            infinite marker (to)
UH	            interjection (goodbye)
VB	            verb (ask)
VBG	            verb gerund (judging)
VBD	            verb past tense (pleaded)
VBN	            verb past participle (reunified)
VBP	            verb, present tense not 3rd person singular(wrap)
VBZ	            verb, present tense with 3rd person singular (bases)
WDT	            wh-determiner (that, what)
WP	            wh- pronoun (who)
WRB	            wh- adverb (how)
'''
