import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import json
import csv
from extra_streamlit_components import tab_bar, TabBarItemData
import matplotlib.pyplot as plt
from datetime import datetime

title = "Jouez avec nous !"
sidebar_name = "Jeu"

@st.cache_data
def init_game():
    new = int(time.time())
    sentence_test = pd.read_csv('../data/multilingue/sentence_test_extract.csv')
    sentence_test = sentence_test[4750:]
    # Lisez le contenu du fichier JSON
    with open('../data/multilingue/lan_to_language.json', 'r') as fichier:
        lan_to_language = json.load(fichier)
    t_now = time.time()
    return sentence_test, lan_to_language, new, t_now

def find_indice(sent_selected):
    l = list(lan_to_language.keys())
    for i in range(len(l)):
        if l[i] == sentence_test['lan_code'].iloc[sent_selected]:
            return i

@st.cache_data
def set_game(new):
    nb_st = len(sentence_test)
    sent_sel = []
    # Utilisez une boucle pour générer 5 nombres aléatoires différents
    while len(sent_sel) < 5:
        nombre = random.randint(0, nb_st)
        if nombre not in sent_sel:
            sent_sel.append(nombre)

    rep_possibles=[]
    for i in range(5):
        rep_possibles.append([find_indice(sent_sel[i])])
        while len(rep_possibles[i]) < 5:
            rep_possible = random.randint(0, 95)
            if rep_possible not in rep_possibles[i]:
                rep_possibles[i].append(rep_possible)
        random.shuffle(rep_possibles[i])
    return sent_sel, rep_possibles, new

def calc_score(n_rep,duration):

    if n_rep==0: return 0
    s1 = n_rep*200
    if duration < 60:
        s2 = (60-duration)*200/60
        if n_rep==5:
            s2 *= 2.5
    else:
        s2 = max(-(duration-60)*100/60,-100)
    s = int(s1+s2)
    return s

def read_leaderboard():
    return pd.read_csv('../data/game_leaderboard.csv', index_col=False,encoding='utf8')

def write_leaderboard(lb):
    lb['Nom'] = lb['Nom'].astype(str)
    lb['Rang'] = lb['Rang'].astype(int)
    lb.to_csv(path_or_buf='../data/game_leaderboard.csv',columns=['Rang','Nom','Score','Timestamp','BR','Duree'],index=False, header=True,encoding='utf8')

def display_leaderboard():
    lb = read_leaderboard()
    st.write("**Leaderboard :**")
    list_champ = """
        | Rang | Nom        | Score |  
        |------|------------|-------|"""
    if len(lb)>0:
        for i in range(len(lb)):
            list_champ += """
        | """+str(lb['Rang'].iloc[i])+""" | """+str(lb['Nom'].iloc[i])[:9]+""" | """+str(lb['Score'].iloc[i])+""" |"""
    st.markdown(list_champ, unsafe_allow_html=True )
    return lb

def write_log(TS,Nom,Score,BR,Duree):
    log = pd.read_csv('../data/game_log.csv', index_col=False,encoding='utf8')
    date_heure = datetime.fromtimestamp(TS)
    Date = date_heure.strftime('%Y-%m-%d %H:%M:%S')
    log = pd.concat([log, pd.DataFrame(data={'Date':[Date], 'Nom':[Nom],'Score':[Score],'BR':[BR],'Duree':[Duree]})], ignore_index=True)
    log.to_csv(path_or_buf='../data/game_log.csv',columns=['Date','Nom','Score','BR','Duree'],index=False, header=True,encoding='utf8')

def display_files():
    log = pd.read_csv('../data/game_log.csv', index_col=False,encoding='utf8')
    lb = pd.read_csv('../data/game_leaderboard.csv', index_col=False,encoding='utf8')
    st.dataframe(lb)
    st.dataframe(log)

def run():
    global sentence_test, lan_to_language

    sentence_test, lan_to_language, new, t_debut = init_game()

    st.write("")
    st.title(title)
    st.write("#### **Etes vous un expert es Langues ?**\n")
    st.markdown(
        """
        Essayer de trouvez, sans aide, la langue des 5 phrases suivantes.  
        Attention : Vous devez être le plus rapide possible !  
        """, unsafe_allow_html=True
        )
    st.write("")
    player_name = st.text_input("Quel est votre nom ?")
    
    if player_name == 'display_files':
        display_files()
        return

    score = 0
    col1, col2 = st.columns([0.7,0.3])
    with col2:
        lb = display_leaderboard()
    with col1:
        sent_sel, rep_possibles, new = set_game(new)
        answer = [""] * 5
        l = list(lan_to_language.values())
        for i in range(5):
            answer[i] = st.radio("**:blue["+sentence_test['sentence'].iloc[sent_sel[i]]+"]**\n",[l[rep_possibles[i][0]],l[rep_possibles[i][1]],l[rep_possibles[i][2]], \
                                                                                        l[rep_possibles[i][3]],l[rep_possibles[i][4]]], horizontal=True, key=i)
        t_previous_debut = t_debut
        t_debut = time.time()
        
        if st.button(label="Valider", type="primary"):
            st.cache_data.clear()

            nb_bonnes_reponses = 0
            for i in range(5):
                if lan_to_language[sentence_test['lan_code'].iloc[sent_sel[i]]]==answer[i]:
                    nb_bonnes_reponses +=1
            
            t_fin = time.time()
            duration = t_fin - t_previous_debut

            score = calc_score(nb_bonnes_reponses,duration)
            write_log(time.time(),player_name,score,nb_bonnes_reponses,duration)
            if nb_bonnes_reponses >=4:
                st.write(":red[**Félicitations, vous avez "+str(nb_bonnes_reponses)+" bonnes réponses !**]")
                st.write(":red[Votre score est de "+str(score)+" points]")
            else:
                if nb_bonnes_reponses >1 : s="s" 
                else: s=""
                st.write("**:red[Vous avez "+str(nb_bonnes_reponses)+" bonne"+s+" réponse"+s+".]**")
                if nb_bonnes_reponses >0 : s="s"
                else: s=""
                st.write(":red[Votre score est de "+str(score)+" point"+s+"]")

            st.write("Bonne réponses:")
            for i in range(5):
                st.write("- "+sentence_test['sentence'].iloc[sent_sel[i]]+" -> :blue[**"+lan_to_language[sentence_test['lan_code'].iloc[sent_sel[i]]]+"**]")
                new = int(time.time())
            st.button(label="Play again ?", type="primary")

            with col2:
                now = time.time()
                # Si le score du dernier est plus vieux d'une semaine, il est remplacé par un score + récent
                renew_old = ((len(lb)>9) and (lb['Timestamp'].iloc[9])<(now-604800)) 
                 
                if (score>0) and ((((score >= lb['Score'].min()) and (len(lb)>9)) or (len(lb)<=9)) or (pd.isna(lb['Score'].min())) or renew_old):
                    if player_name not in lb['Nom'].tolist():
                        if (((score >= lb['Score'].min()) and (len(lb)>9)) or (len(lb)<=9)) or (pd.isna(lb['Score'].min())) :
                            lb = pd.concat([lb, pd.DataFrame(data={'Nom':[player_name],'Score':[score],'Timestamp':[now],'BR':[nb_bonnes_reponses],'Duree':[duration]})], ignore_index=True)
                            lb = lb.sort_values(by=['Score', 'Timestamp'], ascending=[False, False]).reset_index()
                            lb = lb.drop(lb.index[10:])
                        else:
                            st.write('2:',player_name)
                            lb['Nom'].iloc[9]= player_name
                            lb['Score'].iloc[9]= score
                            lb['Timestamp'].iloc[9]=now
                            lb['BR'].iloc[9]=nb_bonnes_reponses
                            lb['Duree'].iloc[9]=duration
                            lb = lb.reset_index()
                    else:
                        liste_Nom = lb['Nom'].tolist()
                        for i,player in enumerate(liste_Nom):
                            if player == player_name:
                                if lb['Score'].iloc[i] < score:
                                    lb['Score'].iloc[i] = score
                                    lb['Timestamp'].iloc[i]=now
                                lb = lb.sort_values(by=['Score', 'Timestamp'], ascending=[False, False]).reset_index()
                    for i in range(len(lb)):
                        if (i>0):
                            if (lb['Score'].iloc[i]==lb['Score'].iloc[i-1]):
                                lb['Rang'].iloc[i] = lb['Rang'].iloc[i-1]
                            else:
                                lb['Rang'].iloc[i] = i+1
                        else:
                            lb['Rang'].iloc[i] = i+1
                    if player_name !="":
                        write_leaderboard(lb)
                
        
    return


        





    
