#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:41:28 2022

@author: nloreedwards
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


output_file = "sic_matches.csv"
sic_codes = pd.read_csv("sic_descriptions.csv", dtype=str)

sic_codes["Three Digit Code"] = sic_codes["sic"].str[0:3]
sic_codes["Three Digit Code"] = "'" + sic_codes["Three Digit Code"] + "'"

sic_codes['description'] = sic_codes.groupby(['Three Digit Code'])['description'].transform(lambda x : ' '.join(x))

sic_codes.drop_duplicates(subset=["Three Digit Code"], keep='first')

Orbis_data = 'missing_sics_descriptions.csv'
orbis = pd.read_csv(Orbis_data)
orbis = orbis[orbis["bvdid"].notnull()]

orbis["company_description"] = orbis["trade_description"].fillna('') + orbis["products_services"].fillna('')

descriptions = sic_codes["description"]
company_description = orbis[orbis["company_description"].notnull()]
company_description = company_description["company_description"]

company_description_df = pd.DataFrame(data=company_description)
descriptions_df = pd.DataFrame(data=descriptions)

all_data = pd.DataFrame(data=company_description)
all_data = all_data.rename(columns={"specialities": "Description"})
all_data = all_data.append(descriptions_df)

s_indices = pd.Series(company_description.index, index=company_description)

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
model = tfidf_vectorizer.fit(descriptions)
descriptions_matrix = tfidf_vectorizer.transform(descriptions)
company_description_matrix = tfidf_vectorizer.transform(company_description)

print(descriptions_matrix.shape)
print(company_description_matrix.shape)

# compute and print the cosine similarity matrix
cosine_sim = linear_kernel(company_description_matrix, descriptions_matrix, dense_output=False)
print(cosine_sim.shape)

scores = []
company_descriptions = []
matched_sic = []

for i in range(cosine_sim.shape[0]):
    
    #print(i)
    company_description_name = company_description_df["company_description"].iloc[i]
    idx = s_indices[company_description_name]
    company_descriptions.append(company_description_name)
    
    row = cosine_sim.getrow(i).A
    
    scores.append(row.max())
    sic_index = row.argmax()
    
    code = sic_codes["Three Digit Code"].iloc[sic_index]
    
    matched_sic.append(code)
    
data = {'company_description': company_descriptions, 'SIC Code': matched_sic, 'SimilScore': scores}
matches = pd.DataFrame(data)

matched_merged = pd.merge(orbis, matches, on="company_description", how="left")
matched_merged = matched_merged.drop_duplicates()

matched_merged.to_csv(output_file, index=False)