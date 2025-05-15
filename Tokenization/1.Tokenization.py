import numpy as np
import pandas as pd
import streamlit as st
import nltk 
from nltk.tokenize import word_tokenize,sent_tokenize,blankline_tokenize,TweetTokenizer,TreebankWordDetokenizer,PunktSentenceTokenizer

st.set_page_config(page_title='NLTK Tokenization',layout='wide')
st.title('NLTK Tokenization')

def apply_tokenization(input_text,mode):
    if mode == 'Word':
        tokens = word_tokenize(input_text)
    elif mode == 'Sentence':
        tokens = sent_tokenize(input_text)
    elif mode == 'Blank':
        tokens = blankline_tokenize(input_text)
    elif mode == 'Tweet':
        tknzr = TweetTokenizer(strip_handles=True,reduce_len=True)
        tokens = tknzr.tokenize(input_text)
    elif mode == 'Punct':
        punckt=PunktSentenceTokenizer()
        tokens = punckt.tokenize(input_text)
    else:
        tokens =[]
    
    if input_text.strip():
        st.subheader('Tokens:')
        st.write(tokens)
        detokenized = TreebankWordDetokenizer().detokenize(tokens)
        st.subheader('Detokenized {Roundtrip}:')
        st.write(detokenized)
    else :
        st.warning('Please enter some text to tokenize')

test_sentences = [
        "When considering a home loan, understanding the long-term cost is essential."
        "For example, on a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88. "
        "Over three decades, this accumulates significantly—highlighting the importance of interest rates.",
        "“We beat some pretty good teams to get here,” Slocum said with a proud grin. "
        "His players surrounded him, cheering and laughing. "
        "It was a hard-earned victory and a moment none of them would forget.",
        "Sometimes television can be painfully predictable. "
        "Well, we couldn't have this cliché-ridden, Touched by an Angel-style wannabe if she didn’t experience a miraculous turnaround. "
        "\"A show creator John Masius worked on,\" someone reminded us—as if that justified the emotional manipulation.",
        "Workplace stress is real, and sometimes it spills out dramatically. "
        "“I cannot cannot work under these conditions!” she shouted, slamming the door behind her. "
        "Everyone froze. Silence followed. Then, the slow clatter of keyboards resumed.",
        "Corporate expenditures can be staggering when broken down. "
        "The company spent 40.75% of its income last year on advertising alone. "
        "Stakeholders raised eyebrows at the figure, prompting a call for budget restructuring.",
        "Punctuality was never his strong suit, but today he tried. "
        "He arrived at 3:00 pm sharp, nervously checking his watch. "
        "Despite his efforts, the meeting had already begun five minutes earlier.",
        "Back from the store, I reviewed my purchases. "
        "I bought these items: books, pencils, and pens. "
        "It wasn’t much, but it covered the essentials for the week.",
        "Crowd turnout exceeded expectations at the rally. "
        "There were 300,000, but that wasn't enough for the organizers. "
        "They had aimed for half a million to make a stronger statement.",
        "She greeted me with a cheerful 'Hello!' and asked, 'How \"are\" you?' "
        "The mix of sincerity and sarcasm threw me off. "
        "Sometimes, words wrapped in quotes carry heavier weight than intended.",
        "<A sentence> with (many) [kinds] of {parentheses}. "
        "\"Sometimes it's inside (quotes).\" "
        "And sometimes, it's the other way around: (\"Sometimes the otherway around\").",
        "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--",
        "RT @facugambande: Ya por arrancar a grabar !!! #TirenTirenTiren vamoo !!",
        "@crushinghes the summer holidays are great but I'm so bored already :(",
        "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        "@jrmy: I'm REALLY HAPPYYY about that! NICEEEE :D :P"
        "@_willy65: No place for @chuck tonight. Sorry."
]

st.sidebar.title('Apply filters')
option = st.sidebar.radio('Select input text',['Select from sentence','Provide your own sentence'])

if option == 'Select from sentence' :
    input_text=st.radio('Choose',test_sentences)
else:
    input_text = st.text_area('Enter your text here',height=150)

if st.button('Word tokenize'):
    apply_tokenization(input_text,'Word')
elif st.button('Sentence Tokenize'):
    apply_tokenization(input_text,'Sentence')
if st.button('Blank Tokenizer'):
    apply_tokenization(input_text,'Blank')
if st.button("Tweet Tokenize"):
    apply_tokenization(input_text, "Tweet")
if st.button("PunctuationSentence Tokenize"):
    apply_tokenization(input_text, "Punct")