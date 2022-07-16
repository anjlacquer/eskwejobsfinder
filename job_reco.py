from PIL import Image

from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st
import streamlit.components.v1 as components
import joblib

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lime import lime_text
from matplotlib import pyplot as plt

import nltk
import numpy as np
import pandas as pd
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH

from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import joblib
from sklearn.metrics.pairwise import cosine_similarity

# created python script for data cleaning in streamlit
from data_cleaning import data_cleaning_jobpost

# def load_model(fname):
#     """
#         fname: path/filename.pkl
#         Loads a model
#     """
#     file = open(fname, 'rb')
#     data = pickle.load(file)
#     file.close()

#     return data

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Actual Page
my_page = st.sidebar.radio('Page Navigation', ['eskweJOBS Finder', 'Job Market Overview', 'About the Project'])


if my_page == 'eskweJOBS Finder':

    image = Image.open('img/eskwejobs.PNG')
    st.image(image)

    input_text = st.text_area('Write your skills and competencies below.', help='You can copy and paste relevant parts of your resume.')

    if st.button("Get my results!"):

        clean_data = data_cleaning_jobpost(sample_text=input_text)
        clean_text = clean_data.generate_clean_text()
        # st.write(clean_desc) ### cleaned description

        vocab_tf = joblib.load('tfidf_vocab_streamlit.pkl')
        loaded_model = joblib.load('sgd_model.sav')
        tf = TfidfVectorizer(stop_words = "english", lowercase = True, max_features = 500000, vocabulary = vocab_tf)

        model_input = tf.fit_transform([clean_text])
        y_pred2 = loaded_model.predict(model_input)

        def get_specialization(input_text, position):
        
            df = pd.read_csv('data/cleaned-data.csv')
            
            df_ = df[df['position'] != 'multiple']
            
            corpus = df_['clean_desc'].values.tolist()
            corpus.append(input_text)

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            text_sim = cosine_sim[len(df_)][:len(df_)] # get similarities of input with others and then remove the similarity with itself
            cos_sim = pd.DataFrame({'cosine_sim': text_sim})
            
            df_cos = df_.join(cos_sim)
            df_cos = df_cos[['position','specialization', 'cosine_sim']]
            
            result = df_cos.sort_values(by=["cosine_sim"], ascending = False)
            result = result[result['position'] == position][['specialization', 'cosine_sim']].drop_duplicates(['specialization'])
            
            pos_ = result['specialization'].values.tolist()
            sim_ = result['cosine_sim'].values.tolist()
            
            arr = []
            for i in range(len(pos_)):
                _pos = pos_[i]
                _sim = sim_[i]
                
                if i == (len(pos_)-1):
                    sent = _pos + ' (' + str(np.round(_sim, 2)) + ')'
                else:
                    sent = _pos + ' (' + str(np.round(_sim, 2)) + ')' + ','
                arr.append(sent)
            
            sentence = " ".join(arr)
            return sentence 
        
        st.markdown("---")
        st.subheader("These are the specializations you may consider and their similarity scores:")
        st.info(get_specialization(clean_text, y_pred2[0])) 
        st.caption("How to read the results: Results are ranked by level of similarity. The higher the similarity score, the closer your current skills are to the specialization's requirements. \nThe specializations were derived from analyzing recent job post requirements for data positions.")

        st.markdown("---")
        if y_pred2[0] == 'data analyst':
            c1, c2 = st.columns((1, 1))

            with c2:
                st.subheader("Topics to explore:")
                st.markdown("""**(1) Data Analyst (General) |** Support company through data analysis of business operations. Regularly analyze, prepare reports, and provide solutions based on data. Knowledgeable in project management and working with teams.""")
                st.markdown("""**(2) Expert Finance Analyst |** Skilled data analyst for the finance industry. Knowledgeable in Finance and/or Accounting. Preferrably with Masters. Proficient in Excel. Fluency in Mandarin. """)
                st.markdown("""**(3) Customer Segmentation |** Analysis of customers through data modelling. Proficiency in Power BI.""")
                st.markdown("""**(4) Thai Data Analyst - Remote Work |** Work remotely as a data analyst for Thailand accounts. Proficiency in English. Communication through Skype.""")
                st.markdown("""**(5) Insurance Industry |** Help team needs on risk analysis Skilled in research and analysis.""")
                st.markdown("""**(6) Fraud Analysis in Food Deliveries |** Detect fraud analysis in a food delivery company """)
                st.markdown("""**(7) University & Education |** Data analysis for the education industry. Knowlegdeable in higher education industry (quality, standards, accreditation, compliance, mapping trends) . Managerial role. """)
                st.markdown("""**(8) Data Analytics as a Service - BPO |** Recruitment for BPOs offering Data Analytics as a Service. Skilled in data visualization.""")
                st.markdown("""**(9) Foreign Languages |** Proficiency in Japanese""")
            
            with c1:
                image = Image.open('img/man2.png')
                st.image(image)

        elif y_pred2[0] == 'data engineer':

            c1, c2 = st.columns((1, 1))

            with c2:
                st.subheader("Topics to explore:")
                st.markdown("""**(1) Experienced Data Engineer (General) |** Requires skills in project management and general data engineering skills. Knowledgeable on how to apply data engineering for business solutions. """)
                st.markdown("""**(2) Experienced Cloud Engineer |**   Build data pipelines and databases. Proficiency in cloud storage for business optimization.""")
                st.markdown("""**(3) Data Engineering Lead On-Site |** Experienced data engineers needed to lead on-site. Communicates with stakeholders and responsible for company's data engineering maintenance, tools, and assets. Proficiency in leadership (team management, communication, project management, business)""")
                st.markdown("""**(4) Pioneer Team in Company |**  Needs experienced data engineers to introduce ETL in business. Offers an opportunity to grow with the company. Proficiency in Python""")
                st.markdown("""**(5) Cloud Architect |** Design company's cloud system (data pipelines & databases). Highly skilled in cloud technology development. Proficiency in Azure, SQL, and Python.""")
            
            with c1:
                image = Image.open('img/man2.png')
                st.image(image)


        elif y_pred2[0] == 'data scientist':

            c1, c2 = st.columns((1, 1))

            with c2:
                st.subheader("Topics to explore:")
                st.markdown("""**(1) Advanced Machine Learning |** Experienced data scientists needed to provide machine learning solutions to the company. Requires skills in project management and working in teams. Proficiency in Machine Learning and Python. """)
                st.markdown("""**(2) Machine Learning (General) |** Build ML models to provide solutions on business processes and/or customer relationships. Must be knowledegable in different tools and complex analysis. Willing to work in teams. """)
                st.markdown("""**(3) Data Science as a Service (DSaaS) |** Skilled in drawing insights from advanced analytics for client needs. Knowledgeable in statistics and insighting.""")
            with c1:
                image = Image.open('img/man2.png')
                st.image(image)


elif my_page == 'Job Market Overview':

        ### fill with LDA ###

    job_pos = st.sidebar.selectbox('Job Families:', ['Data Analyst', 'Data Engineer', 'Data Scientist'])

    if job_pos == 'Data Analyst':

        mystyle = '''
            <style>
                p {
                    text-align: justify;
                }
            </style>
            '''

        st.markdown(mystyle, unsafe_allow_html=True)

        image = Image.open('img/head_da.PNG')
        st.image(image)
        st.markdown("---")
        # Generate Three equal columns
        c1, c2 = st.columns((1, 1))

        with c1:
            st.subheader("Topics:")
            st.markdown("""**(1) Data Analyst (General) |** Support company through data analysis of business operations. Regularly analyze, prepare reports, and provide solutions based on data. Knowledgeable in project management and working with teams.""")
            st.markdown("""**(2) Expert Finance Analyst |** Skilled data analyst for the finance industry. Knowledgeable in Finance and/or Accounting. Preferrably with Masters. Proficient in Excel. Fluency in Mandarin. """)
            st.markdown("""**(3) Customer Segmentation |** Analysis of customers through data modelling. Proficiency in Power BI.""")
            st.markdown("""**(4) Thai Data Analyst - Remote Work |** Work remotely as a data analyst for Thailand accounts. Proficiency in English. Communication through Skype.""")
            st.markdown("""**(5) Insurance Industry |** Help team needs on risk analysis Skilled in research and analysis.""")
            st.markdown("""**(6) Fraud Analysis in Food Deliveries |** Detect fraud analysis in a food delivery company """)
            st.markdown("""**(7) University & Education |** Data analysis for the education industry. Knowlegdeable in higher education industry (quality, standards, accreditation, compliance, mapping trends) . Managerial role. """)
            st.markdown("""**(8) Data Analytics as a Service - BPO |** Recruitment for BPOs offering Data Analytics as a Service. Skilled in data visualization.""")
            st.markdown("""**(9) Foreign Languages |** Proficiency in Japanese""")

        with c2:
            image = Image.open('img/wc_da.png')
            st.image(image)
            
        st.markdown("---")
        html_file = open('lda/lda-da.html', 'r', encoding='utf-8')
        source = html_file.read()

        components.html(source, height=1000, width=1500)
        ### fill with LDA data analyst

    elif job_pos == 'Data Engineer':

        mystyle = '''
            <style>
                p {
                    text-align: justify;
                }
            </style>
            '''
            
        st.markdown(mystyle, unsafe_allow_html=True)

        image = Image.open('img/head_de.PNG')
        st.image(image)
        st.markdown("---")

        c1, c2 = st.columns((1, 1))

        with c2:
            image = Image.open('img/wc_de.png')
            st.image(image)

        with c1:
            st.subheader("Topics:")
            st.markdown("""**(1) Experienced Data Engineer (General) |** Requires skills in project management and general data engineering skills. Knowledgeable on how to apply data engineering for business solutions. """)
            st.markdown("""**(2) Experienced Cloud Engineer |**   Build data pipelines and databases. Proficiency in cloud storage for business optimization.""")
            st.markdown("""**(3) Data Engineering Lead On-Site |** Experienced data engineers needed to lead on-site. Communicates with stakeholders and responsible for company's data engineering maintenance, tools, and assets. Proficiency in leadership (team management, communication, project management, business)""")
            st.markdown("""**(4) Pioneer Team in Company |**  Needs experienced data engineers to introduce ETL in business. Offers an opportunity to grow with the company. Proficiency in Python""")
            st.markdown("""**(5) Cloud Architect |** Design company's cloud system (data pipelines & databases). Highly skilled in cloud technology development. Proficiency in Azure, SQL, and Python.""")
            
        st.markdown("---")
        html_file = open('lda/lda-de.html', 'r', encoding='utf-8')
        source = html_file.read()

        components.html(source, height=1000, width=1500)

        ### fill with LDA data analyst

    elif job_pos == 'Data Scientist':

        mystyle = '''
            <style>
                p {
                    text-align: justify;
                }
            </style>
            '''
            
        st.markdown(mystyle, unsafe_allow_html=True)

        image = Image.open('img/head_ds.PNG')
        st.image(image)
        st.markdown("---")

        # Generate Three equal columns
        c1, c2 = st.columns((1, 1))

        with c2:
            image = Image.open('img/wc_ds.png')
            st.image(image)

        with c1:
            st.subheader("Topics:")
            st.markdown("""**(1) Advanced Machine Learning |** Experienced data scientists needed to provide machine learning solutions to the company. Requires skills in project management and working in teams. Proficiency in Machine Learning and Python. """)
            st.markdown("""**(2) Machine Learning (General) |** Build ML models to provide solutions on business processes and/or customer relationships. Must be knowledegable in different tools and complex analysis. Willing to work in teams. """)
            st.markdown("""**(3) Data Science as a Service (DSaaS) |** Skilled in drawing insights from advanced analytics for client needs. Knowledgeable in statistics and insighting.""")

        st.markdown("---")
        html_file = open('lda/lda-ds.html', 'r', encoding='utf-8')
        source = html_file.read()

        components.html(source, height=1000, width=1500)

        ### fill with LDA data analyst
    
elif my_page == 'About the Project':

        st.title("About the Project")
        def page_slide():
            components.html(
        '<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT0BSqCK2IBpGqnPPJ2WkPTTMS3wSfjqTkBXRHz7ccYGrEyaatJ5EmCj9f_PPMACItZeyf1xUmyob5M/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>' 
            ,height=1080, width=1920)

        page_slide()

    
        ### fill with slides ###
    
## Credits
st.markdown("---")
st.sidebar.markdown("""
### The Team
- Karl Aleta
- Ely Geniston
- Anj Lacar
- Gelo Maandal

Mentored by Marco Francisco                                                                     
Taught by Rhey Ann Magcalas                                                                   
#### Eskwelabs Data Science Fellows Cohort 9 
""")