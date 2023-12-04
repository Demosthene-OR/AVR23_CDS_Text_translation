import streamlit as st
import os
import numpy as np
import pandas as pd
import collections
from nltk.tokenize import word_tokenize
from nltk import download
from ast import literal_eval
from translate_app import tr
if st.session_state.Cloud == 0:
    import nltk
    import contextlib
    import re
    from nltk.corpus import stopwords
    import warnings
    warnings.filterwarnings('ignore')
# from PIL import Image
# import time
# import random

title = "Exploration et Preprocessing"
sidebar_name = "Exploration et Preprocessing"
dataPath = st.session_state.DataPath

# Indiquer si l'on veut enlever les stop words. C'est un processus long
stopwords_to_do = True
# Indiquer si l'on veut lemmatiser les phrases, un fois les stop words enlevés. C'est un processus long (approximativement 8 minutes)
lemmatize_to_do = True
# Indiquer si l'on veut calculer le score Bleu pour tout le corpus. C'est un processus très long long (approximativement 10 minutes pour les 10 dictionnaires)
bleu_score_to_do = True
# Première ligne à charger
first_line = 0
# Nombre maximum de lignes à charger
max_lines = 140000
if ((first_line+max_lines)>137860):
    max_lines = max(137860-first_line ,0)
# Nombre maximum de ligne à afficher pour les DataFrame
max_lines_to_display = 50

download('punkt')

if st.session_state.Cloud == 0:
    download('averaged_perceptron_tagger')
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        download('stopwords')

@st.cache_data
def load_data(path):
    
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        
    # On convertit les majuscules en minulcule
    data = data.lower()
    data = data.split('\n')
    return data[first_line:min(len(data),first_line+max_lines)]

@st.cache_data
def load_preprocessed_data(path,data_type):
    
    input_file = os.path.join(path)
    if data_type == 1:
        return pd.read_csv(input_file, encoding="utf-8", index_col=0)
    else:
        with open(input_file, "r",  encoding="utf-8") as f:
            data = f.read()
            data = data.split('\n')
        if data_type==0:
            data=data[:-1]
        elif data_type == 2:
            data=[eval(i) for i in data[:-1]]
        elif data_type ==3:
            data2 = []
            for d in data[:-1]:
                data2.append(literal_eval(d))
            data=data2
        return data

@st.cache_data
def load_all_preprocessed_data(lang):
    txt             =load_preprocessed_data(dataPath+'/preprocess_txt_'+lang,0)
    txt_split       = load_preprocessed_data(dataPath+'/preprocess_txt_split_'+lang,3)
    txt_lem         = load_preprocessed_data(dataPath+'/preprocess_txt_lem_'+lang,0)
    txt_wo_stopword = load_preprocessed_data(dataPath+'/preprocess_txt_wo_stopword_'+lang,0)
    df_count_word   = pd.concat([load_preprocessed_data(dataPath+'/preprocess_df_count_word1_'+lang,1), load_preprocessed_data(dataPath+'/preprocess_df_count_word2_'+lang,1)]) 
    return txt, txt_split, txt_lem, txt_wo_stopword, df_count_word

#Chargement des textes complet dans les 2 langues
full_txt_en = load_data(dataPath+'/small_vocab_en')
full_txt_fr = load_data(dataPath+'/small_vocab_fr')

# Chargement du résultat du préprocessing, si st.session_state.reCalcule == False
if not st.session_state.reCalcule:
    full_txt_en, full_txt_split_en, full_txt_lem_en, full_txt_wo_stopword_en, full_df_count_word_en = load_all_preprocessed_data('en')
    full_txt_fr, full_txt_split_fr, full_txt_lem_fr, full_txt_wo_stopword_fr, full_df_count_word_fr = load_all_preprocessed_data('fr')
else:

    def remove_stopwords(text, lang): 
        stop_words = set(stopwords.words(lang))
        # stop_words will contain  set all english stopwords
        filtered_sentence = []   
        for word in text.split(): 
            if word not in stop_words: 
                filtered_sentence.append(word) 
        return " ".join(filtered_sentence)

    def clean_undesirable_from_text(sentence, lang):

        # Removing URLs 
        sentence  = re.sub(r"https?://\S+|www\.\S+", "", sentence )
    
        # Removing Punctuations (we keep the . character)
        REPLACEMENTS = [("..", "."),
                        (",", ""),
                        (";", ""),
                        (":", ""),
                        ("?", ""),
                        ('"', ""),
                        ("-", " "),
                        ("it's", "it is"),
                        ("isn't","is not"),
                        ("'", " ")
                        ]
        for old, new in REPLACEMENTS:
            sentence = sentence.replace(old, new)
    
        # Removing Digits
        sentence= re.sub(r'[0-9]','',sentence)
    
        # Removing Additional Spaces
        sentence = re.sub(' +', ' ', sentence)

        return sentence

    def clean_untranslated_sentence(data1, data2):
        i=0
        while i<len(data1):
            if data1[i]==data2[i]:
                data1.pop(i)
                data2.pop(i)
            else: i+=1
        return data1,data2

    import spacy

    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')


    def lemmatize(sentence,lang):
        # Create a Doc object
        if lang=='en':
            nlp=nlp_en
        elif lang=='fr':
            nlp=nlp_fr
        else: return
        doc = nlp(sentence)

        # Create list of tokens from given string
        tokens = [] 
        for token in doc:
            tokens.append(token)

        lemmatized_sentence = " ".join([token.lemma_ for token in doc])
 
        return lemmatized_sentence


    def preprocess_txt (data, lang): 
  
        word_count = collections.Counter()
        word_lem_count = collections.Counter()
        word_wosw_count = collections.Counter()
        corpus = []
        data_split = []
        sentence_length = []
        data_split_wo_stopwords = []
        data_length_wo_stopwords = []
        data_lem = []
        data_lem_length = []
    
        txt_en_one_string= ". ".join([s for s in data])
        txt_en_one_string = txt_en_one_string.replace('..', '.')
        txt_en_one_string = " "+clean_undesirable_from_text(txt_en_one_string, 'lang')
        data = txt_en_one_string.split('.')
        if data[-1]=="":
            data.pop(-1)
        for i in range(len(data)): # On enleve les ' ' qui commencent et finissent les phrases 
            if data[i][0] == ' ':
                data[i]=data[i][1:]
            if data[i][-1] == ' ':
                data[i]=data[i][:-1]
        nb_phrases = len(data)
    
        # Création d'un tableau de mots (sentence_split)
        for i,sentence in enumerate(data):
            sentence_split = word_tokenize(sentence)
            word_count.update(sentence_split)
            data_split.append(sentence_split)
            sentence_length.append(len(sentence_split))

        # La lemmatisation et le nettoyage des stopword va se faire en batch pour des raisons de vitesse
        # (au lieu de le faire phrase par phrase)
        # Ces 2 processus nécéssitent de connaitre la langue du corpus
        if lang == 'en': l='english'
        elif lang=='fr': l='french'
        else: l="unknown"

        if l!="unknown":
            # Lemmatisation en 12 lots (On ne peut lemmatiser + de 1 M de caractères à la fois)
            data_lemmatized=""
            if lemmatize_to_do:
                n_batch = 12
                batch_size = round((nb_phrases/ n_batch)+0.5)
                for i in range(n_batch):
                    to_lem = ".".join([s for s in data[i*batch_size:(i+1)*batch_size]])
                    data_lemmatized = data_lemmatized+"."+lemmatize(to_lem,lang).lower()

                data_lem_for_sw = data_lemmatized[1:]  
                data_lemmatized = data_lem_for_sw.split('.')
                for i in range(nb_phrases):
                    data_lem.append(data_lemmatized[i].split())
                    data_lem_length.append(len(data_lemmatized[i].split()))
                    word_lem_count.update(data_lem[-1])
                               
            # Elimination des StopWords en un lot
            # On élimine les Stopwords des phrases lémmatisés, si cette phase a eu lieu
            # (wosw signifie "WithOut Stop Words")
            if stopwords_to_do:
                if lemmatize_to_do:
                    data_wosw = remove_stopwords(data_lem_for_sw,l)
                else:
                    data_wosw = remove_stopwords(txt_en_one_string,l)
                               
                data_wosw = data_wosw.split('.')
                for i in range(nb_phrases):
                    data_split_wo_stopwords.append(data_wosw[i].split())
                    data_length_wo_stopwords.append(len(data_wosw[i].split()))
                    word_wosw_count.update(data_split_wo_stopwords[-1])

        corpus = list(word_count.keys())
   
        # Création d'un DataFrame txt_n_unique_val :
        #      colonnes = mots
        #      lignes = phases
        #      valeur de la cellule = nombre d'occurence du mot dans la phrase
    
        ## BOW
        from sklearn.feature_extraction.text import CountVectorizer
        count_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"[^' ']+" )
    
        # Calcul du nombre d'apparition de chaque mot dans la phrases
        countvectors = count_vectorizer.fit_transform(data)
        corpus = count_vectorizer.get_feature_names_out()

        txt_n_unique_val=  pd.DataFrame(columns=corpus,index=range(nb_phrases), data=countvectors.todense()).astype(float)     
    
        return data, corpus, data_split, data_lemmatized, data_wosw, txt_n_unique_val, sentence_length, data_length_wo_stopwords, data_lem_length      
 

def count_world(data):
    word_count = collections.Counter()
    for sentence in data:
        word_count.update(word_tokenize(sentence))
    corpus = list(word_count.keys())
    nb_mots = sum(word_count.values())
    nb_mots_uniques = len(corpus)
    return corpus, nb_mots, nb_mots_uniques
                          
def display_preprocess_results(lang, data, data_split, data_lem, data_wosw, txt_n_unique_val):

    global max_lines, first_line, last_line, lemmatize_to_do, stopwords_to_do
    corpus = []
    nb_phrases = len(data)
    corpus, nb_mots, nb_mots_uniques = count_world(data)
    mots_lem, _ , nb_mots_lem = count_world(data_lem)
    mots_wo_sw, _ , nb_mots_wo_stopword = count_world(data_wosw)
    # Identifiez les colonnes contenant uniquement des zéros et les supprimer
    columns_with_only_zeros = txt_n_unique_val.columns[txt_n_unique_val.eq(0).all()]
    txt_n_unique_val = txt_n_unique_val.drop(columns=columns_with_only_zeros)

    # Affichage du nombre de mot en fonction du pré-processing réalisé
    tab1, tab2, tab3, tab4 = st.tabs([tr("Résumé"), tr("Tokenisation"),tr("Lemmatisation"), tr("Sans Stopword")])
    with tab1:
        st.subheader(tr("Résumé du pré-processing"))
        st.write("**"+tr("Nombre de phrases")+"                     : "+str(nb_phrases)+"**")
        st.write("**"+tr("Nombre de mots")+"                        : "+str(nb_mots)+"**")
        st.write("**"+tr("Nombre de mots uniques")+"                : "+str(nb_mots_uniques)+"**") 
        st.write("") 
        st.write("\n**"+tr("Nombre d'apparitions de chaque mot dans chaque phrase (:red[Bag Of Words]):")+"**")
        st.dataframe(txt_n_unique_val.head(max_lines_to_display), width=800) 
    with tab2:
        st.subheader(tr("Tokenisation"))
        st.write(tr('Texte "splited":'))
        st.dataframe(pd.DataFrame(data=data_split, index=range(first_line,last_line)).head(max_lines_to_display).fillna(''), width=800)
        st.write("**"+tr("Nombre de mots uniques")+"                : "+str(nb_mots_uniques)+"**") 
        st.write("")
        st.write("\n**"+tr("Mots uniques")+":**")
        st.markdown(corpus[:500])
        st.write("\n**"+tr("Nombre d'apparitions de chaque mot dans chaque phrase (:red[Bag Of Words]):")+"**")
        st.dataframe(txt_n_unique_val.head(max_lines_to_display), width=800) 
    with tab3:
        st.subheader(tr("Lemmatisation"))
        if lemmatize_to_do:  
            st.dataframe(pd.DataFrame(data=data_lem,columns=[tr('Texte lemmatisé')],index=range(first_line,last_line)).head(max_lines_to_display), width=800)
            # Si langue anglaise, affichage du taggage des mots
            # if lang == 'en':
            #     for i in range(min(5,len(data))):
            #         s = str(nltk.pos_tag(data_split[i]))
            #         st.markdown("**Texte avec Tags     "+str(i)+"** : "+s)
            st.write("**"+tr("Nombre de mots uniques lemmatisés")+"     : "+str(nb_mots_lem)+"**")
            st.write("")
            st.write("\n**"+tr("Mots uniques lemmatisés:")+"**")
            st.markdown(mots_lem[:500])  
    with tab4:
        st.subheader(tr("Sans Stopword"))
        if stopwords_to_do:
            st.dataframe(pd.DataFrame(data=data_wosw,columns=['Texte sans stopwords'],index=range(first_line,last_line)).head(max_lines_to_display), width=800)
            st.write("**"+tr("Nombre de mots uniques sans stop words")+": "+str(nb_mots_wo_stopword)+"**")
            st.write("")  
            st.write("\n**"+tr("Mots uniques sans stop words")+":**")
            st.markdown(mots_wo_sw[:500])


def run():
    global max_lines, first_line, last_line, lemmatize_to_do, stopwords_to_do
    global full_txt_en, full_txt_split_en, full_txt_lem_en, full_txt_wo_stopword_en, full_df_count_word_en
    global full_txt_fr, full_txt_split_fr, full_txt_lem_fr, full_txt_wo_stopword_fr, full_df_count_word_fr
    
    st.write("")
    st.title(tr(title))
    
    st.write("## **"+tr("Explications")+" :**\n")
    st.markdown(tr(
        """
        Le traitement du langage naturel permet à l'ordinateur de comprendre et de traiter les langues humaines.
        Lors de notre projet, nous avons étudié le dataset small_vocab, proposés par Suzan Li, Chief Data Scientist chez Campaign Research à Toronto.
        Celui-ci représente un corpus de phrases simples en anglais, et sa traduction (approximative) en français. 
        :red[**Small_vocab**] contient 137 860 phrases en anglais et français.  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        Afin de découvrir ce corpus et de préparer la traduction, nous allons effectuer un certain nombre de tâches de pré-traitement (preprocessing).
        Ces taches sont, par exemple:  
        """)
    , unsafe_allow_html=True)
    st.markdown(
        "* "+tr("le :red[**nettoyage**] du texte (enlever les majuscules et la ponctuation)")+"\n"+ \
        "* "+tr("la :red[**tokenisation**] (découpage du texte en mots)")+"\n"+ \
        "* "+tr("la :red[**lemmatisation**] (traitement lexical qui permet de donner une forme unique à toutes les \"variations\" d'un même mot)")+"\n"+ \
        "* "+tr("l'élimination des :red[**mots \"transparents\"**] (sans utilité pour la compréhension, tels que les articles).")+"  \n"+ \
        tr("Ce prétraintement se conclut avec la contruction d'un :red[**Bag Of Worlds**], c'est à dire une matrice qui compte le nombre d'apparition de chaque mots (colonne) dans chaque phrase (ligne)")
    , unsafe_allow_html=True)
    # 
    st.write("## **"+tr("Paramètres")+" :**\n")
    Langue = st.radio(tr('Langue:'),('Anglais','Français'), horizontal=True)
    first_line = st.slider(tr('No de la premiere ligne à analyser:'),0,137859)
    max_lines = st.select_slider(tr('Nombre de lignes à analyser:'),
                              options=[1,5,10,15,100, 500, 1000,'Max'])
    if max_lines=='Max':
        max_lines=137860
    if ((first_line+max_lines)>137860):
        max_lines = max(137860-first_line,0)
        
    last_line = first_line+max_lines
    if (Langue=='Anglais'):
        st.dataframe(pd.DataFrame(data=full_txt_en,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    else:
        st.dataframe(pd.DataFrame(data=full_txt_fr,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    st.write("")

    # Chargement des textes sélectionnés dans les 2 langues (max lignes = max_lines)
    txt_en = full_txt_en[first_line:last_line]
    txt_fr = full_txt_fr[first_line:last_line]
    
    # Elimination des phrases non traduites
    # txt_en, txt_fr = clean_untranslated_sentence(txt_en, txt_fr)

    if not st.session_state.reCalcule:
        txt_split_en = full_txt_split_en[first_line:last_line]
        txt_lem_en = full_txt_lem_en[first_line:last_line]
        txt_wo_stopword_en = full_txt_wo_stopword_en[first_line:last_line]
        df_count_word_en = full_df_count_word_en.loc[first_line:last_line-1]
        txt_split_fr = full_txt_split_fr[first_line:last_line]
        txt_lem_fr = full_txt_lem_fr[first_line:last_line]
        txt_wo_stopword_fr = full_txt_wo_stopword_fr[first_line:last_line]
        df_count_word_fr = full_df_count_word_fr.loc[first_line:last_line-1]

    # Lancement du préprocessing du texte qui va spliter nettoyer les phrases et les spliter en mots 
    # et calculer nombre d'occurences des mots dans chaque phrase
    if (Langue == 'Anglais'):
        st.write("## **"+tr("Préprocessing de small_vocab_en")+" :**\n")
        if max_lines>10000:
            with st.status(":sunglasses:", expanded=True):
                if st.session_state.reCalcule:
                    txt_en, corpus_en, txt_split_en, txt_lem_en, txt_wo_stopword_en, df_count_word_en,sent_len_en, sent_wo_sw_len_en, sent_lem_len_en  = preprocess_txt (txt_en,'en')
                display_preprocess_results('en',txt_en, txt_split_en, txt_lem_en, txt_wo_stopword_en, df_count_word_en)
        else:
            if st.session_state.reCalcule:
                txt_en, corpus_en, txt_split_en, txt_lem_en, txt_wo_stopword_en, df_count_word_en,sent_len_en, sent_wo_sw_len_en, sent_lem_len_en  = preprocess_txt (txt_en,'en')
            display_preprocess_results('en',txt_en, txt_split_en, txt_lem_en, txt_wo_stopword_en, df_count_word_en)
    else:
        st.write("## **"+tr("Préprocessing de small_vocab_fr")+" :**\n")
        if max_lines>10000:
            with st.status(":sunglasses:", expanded=True):
                if st.session_state.reCalcule:
                    txt_fr, corpus_fr, txt_split_fr, txt_lem_fr, txt_wo_stopword_fr, df_count_word_fr,sent_len_fr, sent_wo_sw_len_fr, sent_lem_len_fr  = preprocess_txt (txt_fr,'fr')
                display_preprocess_results('fr', txt_fr, txt_split_fr, txt_lem_fr, txt_wo_stopword_fr, df_count_word_fr)
        else:
            if st.session_state.reCalcule:
                txt_fr, corpus_fr, txt_split_fr, txt_lem_fr, txt_wo_stopword_fr, df_count_word_fr,sent_len_fr, sent_wo_sw_len_fr, sent_lem_len_fr  = preprocess_txt (txt_fr,'fr')
            display_preprocess_results('fr', txt_fr, txt_split_fr, txt_lem_fr, txt_wo_stopword_fr, df_count_word_fr)











