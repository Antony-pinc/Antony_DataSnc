#!/usr/bin/env python
# coding: utf-8

# In[1]:

import subprocess
import sys
#!pip uninstall boto3 --yes
#!pip uninstall s3fs --yes
#!pip uninstall aiobotocore --yes 
#!pip install Cython
#! pip install regex
#! pip install xlsxwriter
#! pip install lorem
#!pip uninstall nmslib --yes
#!pip install --no-binary :all: nmslib
#!pip uninstall spacy-ann-linker --yes
#!pip uninstall spacy --yes
#
#!pip uninstall spacy-ann-linker --yes
#!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.0.0/en_core_web_lg-3.0.0.tar.gz
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
#! pip install en_core_web_lg-3.0.0.tar.gz
#!pip install tqdm
#!pip install bs4
#!pip install markdown
#!pip install spacy==3.0.5
#!pip install negspacy==1.0.2
#!pip install scispacy==0.4.0
#!pip install owlready2
#!pip install fsspec
#!pip install s3fs
#!pip install bs4
#!pip install pyarrow
#!pip install fastparquet
#! pip install openpyxl
#!pip install boto3

subprocess.check_call([sys.executable, "-m", "pip", "install", 'Cython'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'regex'])
#subprocess.check_call([sys.executable, "-m", "pip", "install", 'random'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'xlsxwriter'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'openpyxl'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'tqdm'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'bs4'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'markdown'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'memory_profiler'])

subprocess.check_call([sys.executable, "-m", "pip", "install", 'scispacy==0.4.0'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'negspacy==1.0.2'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'owlready2'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'fsspec'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 's3fs'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'bs4'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'boto3'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'pyarrow'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'fastparquet'])


# In[2]:


# IMPORT ALL UTILITY FUNCTIONS

#loading general packages
import pandas as pd
import numpy as np
import regex as re
from  tqdm import tqdm
import s3fs
#from lorem.text import TextLorem
import random
import openpyxl
import xlsxwriter
#from functools import reduce
#from memory_profiler import profile
#import simplejson as json

from bs4 import BeautifulSoup
from markdown import markdown
import re
#import pyarrow
#import fastparquet
#loading NLP packages
import spacy
import pickle
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.pipeline import merge_entities
import scispacy
#import en_core_web_lg
#import en_core_sci_lg
from spacy.language import Language
from spacy.pipeline import EntityRuler 
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc, Span
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy import displacy
from scispacy.linking import EntityLinker

import srsly


import warnings
warnings.filterwarnings('ignore')

#loading packages for mappings to med terminologies
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *

import multiprocessing
from multiprocessing import Pool

# Importing Utility program
from radminer_utility import *

# Total number of CPU processors
print("Number of processors: ", multiprocessing.cpu_count())


# In[3]:


# Importing TUI
### TUI file has semantic relationship types for UMLS child and parent semantic labels which is mapped through Entity Linker
TUI=pd.read_csv("tui_best.csv") 


# ### Load Ontology

# In[4]:


#getting terminologies through pymedtermino
default_world.set_backend(filename = "pym.sqlite3")
#import_umls("/home/ec2-user/SageMaker/HeadCT/umls-2020AA-full.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
default_world.save()
PYM = get_ontology("http://PYM/").load()
SNOMEDCT_US = PYM["SNOMEDCT_US"]
ICD10 = PYM["ICD10"]
CUI = PYM["CUI"]
#testing connection to ontology
PYM.Concepts([ CUI["C0013604"]]) >> SNOMEDCT_US


# In[5]:


#how we can term objects 
list(CUI["C0848397"] >> SNOMEDCT_US)[0] if len(list(CUI["C0848397"] >> SNOMEDCT_US))>0 else ''
#SNOMEDCT_US[410016009]


# In[6]:


CUI.search("Throat culture")


# In[7]:


ICD10.search("Throat culture")


# In[8]:


SNOMEDCT_US.search("Throat culture")


# ### Building Spacy Pipeline

# In[9]:


nlp = spacy.load('model-best-ner-spacyv3/')
nlp.pipe_names


# ### Negations

# In[10]:


ts = termset("en_clinical_sensitive")
ts.add_patterns({
            "pseudo_negations":['not normal', 'not clear'],

        "preceding_negations" : ["stable","clear","without",'within normal limits','no evidence of','there is no','no suspicious','unremarkable','normal',"mother","no","never",
                                "denies","not", "neg","negative","NEGATIVE","neg.",
                               "father","spouse", "wife" ,"aunt","uncle", "mom", "dad","maternal","paternal","grandfather",
                               "grandmother", "husband", "sister", "brother", "family",'History', "history","fh","fam hx",
                               "hx:","f.hx",".hx", 'nfamily',"pshx","shx", "sh", "psh","girlfriend","boyfriend", "employee",
                                'nfamily',"pshx","shx"],
        "termination" : ["except","nevertheless","absence"],

         "following_negations" : ["stable","clear",'within normal limits','no suspicious','unremarkable','normal',"neg.","denied","denies","neg",
                              "negative","denied","NEGATIVE","not","mother","no","No", "never","not", "neg", 
                              "negative","NEGATIVE","neg.", 
                              "father", "spouse", "wife" ,"aunt","uncle", "mom", "dad", "husband","maternal",
                              "paternal","grandfather","grandmother","sister", "brother","family",'History', 
                              "history","fh","fam hx","hx", "sh", "psh","girlfriend","boyfriend","employee",
                              'nfamily',"pshx","shx"],
        })


# In[11]:


# Import ruler_patters
patter = srsly.read_json("ruler_patterns.json") # ruler patterns
patter[:15]


# ### Building Test-Result & General NLP pipelines

# In[12]:


# Test Result NLP Pipeline

def load_model_result():
    try:
        ent_result_patterns1 = {
                           "ANATOMY": {"patterns": [ [{"ENT_TYPE": "MODIFIER"}],
                                                                         ],"n":4 ,"direction": "both" },
                           "OBSERVATION": {"patterns": [ [{"ENT_TYPE": "MODIFIER"}],
                                                                         ],"n":4 ,"direction": "both" }
                          }

        ent_result_patterns2 = {
                           "ANATOMY": {"patterns": [ [{"ENT_TYPE": "RESULT"}]
                                                                             ],"n":5 ,"direction": "right" } ,
                           "OBSERVATION": {"patterns": [ [{"ENT_TYPE": "RESULT"}]
                                                                             ],"n":5 ,"direction": "right" } 
                          }

        nlp = spacy.load('model-best-ner-spacyv3')

        nlp.add_pipe('sentencizer', before="ner")
        #nlp.add_pipe("merge_entities")
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        ruler.add_patterns(patter) #all_result++gender
        nlp.add_pipe("joinentity")
        #nlp.add_pipe("retokenizer")
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("joinentity_observ_anat")
        nlp.add_pipe("retokenizer",last=True)
        #nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "max_entities_per_mention": 5, 'threshold' : 0.55})
        nlp.add_pipe("valext", config={"ent_patterns":ent_result_patterns1}, last=True)
        nlp.add_pipe("valext2", config={"ent_patterns":ent_result_patterns2}, last=True)
        nlp.add_pipe(
        "negex",
        config={
            "chunk_prefix": ['unremarkable','normal',"mother-","father-","no","No","had","history","not","denies","probably",
                                "anticept","without","negative","hx","mother","no","never","denies","not", "neg", "negative",
                                "NEGATIVE","neg.",
                                "father", "spouse","wife" ,"aunt","uncle", "mom", "dad","maternal","paternal","grandfather",
                                "grandmother", "husband", "sister", "brother", "family", "history","fh","fam hx","hx:","f.hx",
                                ".hx","hx",  "sh", "psh", "girlfriend","boyfriend","employee",'there is no','within normal limits',"stable","clear"],
        },
        last=True,
    )
    except:
        pass
    return nlp


# In[13]:


# General NLP Pipeline


def load_general_model(): 
    try:
        nlp_gen=spacy.load("model-best-ner-spacyv3")
        nlp_gen.add_pipe('sentencizer', before="ner")
        nlp_gen.add_pipe('set_custom_boundaries', before="ner")
        ruler = nlp_gen.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patter)
        nlp_gen.add_pipe("scispacy_linker", config={"linker_name": "umls", "max_entities_per_mention": 5, 'threshold' : 0.55})
        nlp_gen.add_pipe(
        "negex",
        config={
            "chunk_prefix": ['unremarkable','normal',"mother-","father-","no","No","had","history","not","denies","probably",
                                "anticept","without","negative","hx","mother","no","never","denies","not", "neg", "negative",
                                "NEGATIVE","neg.",
                                "father", "spouse","wife" ,"aunt","uncle", "mom", "dad","maternal","paternal","grandfather",
                                "grandmother", "husband", "sister", "brother", "family", "history","fh","fam hx","hx:","f.hx",
                                ".hx","hx",  "sh", "psh", "girlfriend","boyfriend","employee",'there is no','within normal limits',"stable","clear"],
        },
        last=True,
        )
    except:
        pass
    return nlp_gen


# In[14]:


nlp_result = load_model_result()
print("NLP components within Test-Result nlp pipe: ",nlp_result.pipe_names)
nlp_gen=load_general_model()
linker = nlp_gen.get_pipe("scispacy_linker")
print("NLP components within General nlp pipe: ",nlp_gen.pipe_names)


# In[ ]:


# text=markdown_to_text(df.report[0])
# doc=nlp_result(text)
# displacy.render(doc, style="ent", options=options)


# ### Prod Functions

# In[ ]:


#       list(CUI[entity._.kb_ents[0][0]] >> SNOMEDCT_US)[0].name if len(list(CUI[entity._.kb_ents[0][0]] >> SNOMEDCT_US))>0 else '',
#       list(CUI[entity._.kb_ents[0][0]] >> ICD10)[0].name if len(list(CUI[entity._.kb_ents[0][0]] >> ICD10))>0 else '',
#      list(CUI[entity._.kb_ents[0][0]] >> SNOMEDCT_US)[0].name if len(list(CUI[entity._.kb_ents[0][0]] >> SNOMEDCT_US))>0 else '',
#      list(CUI[entity._.kb_ents[0][0]] >> ICD10)[0].name if len(list(CUI[entity._.kb_ents[0][0]] >> ICD10))>0 else '',


# In[ ]:


def general_table_utility(doc,p_key):
    #empty=[{"Entity":"", "Entity_label":"","Is_Negated":"","UMLS_CUI":"","UMLS_Name":"","UMLS_child_label":"","UMLS_parent_label":"","UMLS_Definition":"","UMLS_Span":""}]
    return [{
          'patient_key': p_key,
          "Entity":str(entity.text), #entity text
          "Entity_label":str(entity.label_), #label
          "Is_Negated":str(entity._.negex), # is negated or not
          "UMLS_CUI":str(entity._.kb_ents[0][0]), # first linker 
          "UMLS_Name":str(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name),  
          "UMLS_child_label":str(TUI[TUI['tui'] == linker.kb.cui_to_entity[entity._.kb_ents[0][0]].types[0]].label_child.values[0]),
          "UMLS_parent_label":str(TUI[TUI['tui'] == linker.kb.cui_to_entity[entity._.kb_ents[0][0]].types[0]].label.values[0]),
          "UMLS_Definition":str(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].definition),
          "UMLS_Span":str(entity.sent)
} for entity in doc.ents if entity.label_ not in ["RESULT", "MODIFIER"] and len(entity._.kb_ents)>0]
    #return output if len(output)>0 else empty


# In[ ]:


def result_linker(entitytext, p_key):
    return [
    { 
    'patient_key': p_key,
    "Test_Name": entitytext.text,
    "Modifier_Result": ' '.join([str(item) for item in entitytext._.value_extract]),
    "Numeric_Result": ' '.join([str(item) for item in entitytext._.value_extract2]),
    "Is_Negated": entitytext._.negex,
     #entity.text,
    "UMLS_CUI":entity._.kb_ents[0][0], 
    "UMLS_Name": linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name,  
    "UMLS_child_label":TUI[TUI['tui'] == linker.kb.cui_to_entity[entity._.kb_ents[0][0]].types[0]].label_child.values[0],
    "UMLS_parent_label":TUI[TUI['tui'] == linker.kb.cui_to_entity[entity._.kb_ents[0][0]].types[0]].label.values[0],
    "UMLS_Definition":linker.kb.cui_to_entity[entity._.kb_ents[0][0]].definition,
    "UMLS_Span": str(entitytext.sent)
        } 
        for entity in nlp_gen(str(entitytext.text)).ents if len(entity._.kb_ents)>0]


# In[ ]:


def output_result(sents, column_values):
    listy=[]
    docs=[nlp_result(str(i)) for i in sents]
    for doc in docs:
        [[listy.append(i) for i in result_linker(e, column_values)] for e in doc.ents if e._.value_extract or e._.value_extract2]
    return listy if len(listy)>0 else []
    


# In[ ]:


def main_function_gen_res_table(text, column_values):
    first_doc=nlp_gen(text)
    gen_list=general_table_utility(first_doc, column_values)
    sents=filter_spans([ent.sent for ent in first_doc.ents if ent.label_=="RESULT" or ent.label_=="MODIFIER"])
    result_list=output_result(sents, column_values)
#     generic=pd.DataFrame.from_dict(gen_list).drop_duplicates().reset_index(drop=True)
#     result=pd.DataFrame.from_dict(result_list).drop_duplicates().reset_index(drop=True)
    
    return gen_list, result_list #generic, result


# In[ ]:


############# ........------>>>>>>>>>>>..........WITHOUT MAPPING TO ICD AND SNOMED ############

cols =['patient_key']
nlp_gen.max_length = 135321000
nlp_result.max_length = 135321000
def finalkick(df):
    generic=[]
    result=[]
    for i in tqdm(df.index):
        column_values=str(df.patient_key[i])
        text=markdown_to_text(df.report[i])
        gen, res=main_function_gen_res_table(text, column_values)
        generic.append(gen)
        result.append(res)
    return generic, result #generic, result


# Input Path
import boto3
#s3://pas-biogen-sagemaker/Antony/Preprocessing/Ongoing/
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('pas-biogen-sagemaker')
chunks=[]
for object_summary in my_bucket.objects.filter(Prefix="Antony/Preprocessing/Ongoing2/"):
    if '.pickle' in str(object_summary.key):
        chunks.append(object_summary.key)
chunks=chunks[30:]   
chunks


# In[23]:


import s3fs
import re
from functools import reduce

s3 = s3fs.S3FileSystem(anon=False)


# In[ ]:

count = 0
for chunk in chunks:
    count += 1
    start_time = time.time()
    chunk_id= re.findall(r"\d+",chunk)
    chunk_id = chunk_id[1] + '-' + chunk_id[2] + '-' + chunk_id[3]  
    file_id = "s3://pas-biogen-sagemaker/"+chunk
    with s3.open(file_id,'rb') as f: 
        df = pickle.load(f)
    df.rename(columns = {'note_text' : 'report'}, inplace = True)    
    df=df.drop_duplicates().reset_index(drop=True)
    df["report_length"]=df['report'].str.split().str.len()
    df=df.loc[(df['report_length']>=10) & (df['report_length']< 10000)]
    df=df.drop_duplicates().reset_index(drop=True)
    df=df.reset_index(drop=True)
    #df=df.head(10)

    print(f"Starting multiprocessing for {chunk_id}")
    #multi process
    df_split = np.array_split(df, 15) 
    pool = Pool(processes=15,maxtasksperchild=1000)
    print("second")

    f_res = [(gen,res) for gen, res in pool.imap(finalkick, df_split)] 
    time.sleep(10)
    pool.close()
    pool.join()
    time.sleep(5)

    print(f"Starting post processing for {chunk_id} after multiprocessor")    
    generals2=list(chain.from_iterable(list(chain.from_iterable(f_res[0]))))
    general_output=pd.DataFrame(generals2)    
    results2=list(chain.from_iterable(list(chain.from_iterable(f_res[1]))))
    results_output=pd.DataFrame(results2)
    results_output=results_output.drop_duplicates().reset_index(drop=True)
    general_output=general_output.drop_duplicates().reset_index(drop=True)

    print(f"Mapping RESULT table to ICD and SNOMED for {chunk_id}")
    results_output['SNOMED'] = results_output['UMLS_CUI'].apply(lambda x: list(CUI[x] >> SNOMEDCT_US)[0].name if len(list(CUI[x] >> SNOMEDCT_US))>0 else '')
    results_output['ICD_Code'] = results_output['UMLS_CUI'].apply(lambda x: list(CUI[x] >> ICD10)[0].name if len(list(CUI[x] >> ICD10))>0 else '')
    results_output=results_output.replace(r'^\s*$', "NULL", regex=True).reset_index(drop=True)

    print(f"Mapping GENERIC table to ICD and SNOMED for {chunk_id}")
    general_output['SNOMED'] = general_output['UMLS_CUI'].apply(lambda x: list(CUI[x] >> SNOMEDCT_US)[0].name if len(list(CUI[x] >> SNOMEDCT_US))>0 else '')
    general_output['ICD_Code'] = general_output['UMLS_CUI'].apply(lambda x: list(CUI[x] >> ICD10)[0].name if len(list(CUI[x] >> ICD10))>0 else '')
    general_output=general_output.replace(r'^\s*$', "NULL", regex=True).reset_index(drop=True)  

    print(f"Start sending output of ----> {chunk_id} to S3")
    s3 = s3fs.S3FileSystem(anon=False)


    #Output Path
    with s3.open(f's3://pas-biogen-sagemaker/Antony/Ongoing_Output/General/{chunk_id}_generic.csv','w') as f: 
        general_output.to_csv(f, index=False)    
    with s3.open(f's3://pas-biogen-sagemaker/Antony/Ongoing_Output/Result/{chunk_id}_result.csv','w') as f: 
        results_output.to_csv(f, index=False)
    with s3.open(f's3://pas-biogen-sagemaker/Antony/Ongoing_Output/Raw_File/{chunk_id}_raw.csv','w') as f: 
        df.to_csv(f, index=False)
    #print(f"Outputs are sent to S3 for ----> {chunk_id}")    
    print(count, "- ", chunk_id, "Time Taken - ",((time.time() - start_time) / 60)," min",flush =True) 

    #del generals
    #del results
    del df
    del df_split
    del results_output
    del general_output
    print(f"Done itaration for ------> {chunk_id}")

