#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:20:26 2022

@author: nloreedwards
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

sic_list = np.loadtxt("/export/home/dor/nloreedwards/Documents/WMS/sic_list.csv", dtype=str)
description = []

for sic in sic_list:
    
    URL = "https://www.osha.gov/sic-manual/" + sic
    print(URL)
    
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")

    results = soup.find(role="main")
    #print(results.prettify())

    job_elements = results.find_all("div", class_="content")
    
    for job_element in job_elements:
        
        company_element = job_element.find("div", class_="field field--name-body field--type-text-with-summary field--label-hidden field--item")
        text = company_element.text.strip()
    
    description.append(text)
    
data = {'sic': sic_list, 'description': description}
sic_descriptions = pd.DataFrame(data)

sic_descriptions.to_csv("sic_descriptions.csv", index=False)