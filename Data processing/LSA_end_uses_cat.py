# -*- coding: utf-8 -*-
"""
Created on Sat May 20 09:38:51 2023

@author: julia

Graphs for the local sensitivity analyses: variations concerning the end-use categories
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


sns.set(font_scale=2)
sns.set_style("ticks")

years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

#to replace
folder_low = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_4_mob_pub_max_2040/"
subfolder_low = "output_4_mob_pub_max_2040/"
folder_nom = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_base_case/"
subfolder_nom = "output_base_case/"
folder_high = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_5_mob_pub_max_2045/"
subfolder_high = "output_5_mob_pub_max_2045/"
folders = [folder_low, folder_nom, folder_high]
subfolders = [subfolder_low, subfolder_nom, subfolder_high]
#%%
cases = ["low", "nom", "high"]
dicofdf_pass = {}
dicofdf_freight = {}
dicofdf_elec = {}
dicofdf_HT = {}
dicofdf_LTdec = {}
dicofdf_LTdhn = {}
for c in range(len(cases)) : 
    df = pd.read_excel(folders[c]+subfolders[c]+"mob_share.xlsx").set_index("Tech")
    df_elec = pd.read_excel(folders[c]+subfolders[c]+"elec_share.xlsx").set_index("Tech")
    df_HT = pd.read_excel(folders[c]+subfolders[c]+"HT_share.xlsx").set_index("Tech")
    df_LTdec = pd.read_excel(folders[c]+subfolders[c]+"LTdhn_share.xlsx").set_index("Tech")
    df_LTdhn = pd.read_excel(folders[c]+subfolders[c]+"LTdec_share.xlsx").set_index("Tech")
    lst = []
    for i in range(len(df)) : 
        obj = df.iloc[i]
        todel = True
        for j in range(len(obj.values)) : 
            if obj.values[j] > 1 : 
                todel = False
        if todel : 
            lst.append(i)        
    df = df.drop(df.index[lst])
    lst_freight = []
    lst_pass = []
    for i in range(len(df)) : 
        name = df.index[i]
        if "FREIGHT" in name or "TRUCK" in name :
            lst_freight.append(i)
        else : 
            lst_pass.append(i)
    df_freight = df.drop(df.index[lst_pass])
    df_pass = df.drop(df.index[lst_freight])
    
    lst_elec = []
    for i in range(len(df_elec)) : 
        obj = df_elec.iloc[i]
        todel = True
        for j in range(len(obj.values)) : 
            if obj.values[j] > 1 : 
                todel = False
        if todel : 
            lst_elec.append(i)        
    df_elec = df_elec.drop(df_elec.index[lst_elec])
    
    interesting = [' NUCLEAR ', " CCGT ", " IND_COGEN_GAS ", " WIND_ONSHORE ", " WIND_OFFSHORE ", " PV ", " CCGT_AMMONIA "]
    lst_interest = []
    others = [0, 0, 0, 0, 0, 0, 0]
    for y in range(len(years)) : 
        for j in range(len(df_elec)) : 
            name = df_elec.index[j]
            if name not in interesting : 
                others[y] += df_elec.iloc[j,y]
                lst_interest.append(j)
    df_elec = df_elec.drop(df_elec.index[lst_interest])
    df_elec.loc[len(df_elec)] = others
    df_elec.rename(index={len(df_elec)-1 : "OTHERS"}, inplace = True)
    
    lst_HT = []
    for i in range(len(df_HT)) : 
        obj = df_HT.iloc[i]
        todel = True
        for j in range(len(obj.values)) : 
            if obj.values[j] > 1 : 
                todel = False
        if todel : 
            lst_HT.append(i)        
    df_HT = df_HT.drop(df_HT.index[lst_HT])
    
    lst_LTdec = []
    for i in range(len(df_LTdec)) : 
        obj = df_LTdec.iloc[i]
        todel = True
        for j in range(len(obj.values)) : 
            if obj.values[j] > 1 : 
                todel = False
        if todel : 
            lst_LTdec.append(i)        
    df_LTdec = df_LTdec.drop(df_LTdec.index[lst_LTdec])
    
    lst_LTdhn = []
    for i in range(len(df_LTdhn)) : 
        obj = df_LTdhn.iloc[i]
        todel = True
        for j in range(len(obj.values)) : 
            if obj.values[j] > 1 : 
                todel = False
        if todel : 
            lst_LTdhn.append(i)        
    df_LTdhn = df_LTdhn.drop(df_LTdhn.index[lst_LTdhn])
    
    total_freight = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_freight)) : 
            total_freight[j] += df_freight.iloc[i,j]
    df_freight.loc[len(df_freight)] = total_freight
    df_freight.rename(index={len(df_freight)-1 : "Total"}, inplace = True)
    
    total_pass = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_pass)) : 
            total_pass[j] += df_pass.iloc[i,j]
    df_pass.loc[len(df_pass)] = total_pass
    df_pass.rename(index={len(df_pass)-1 : "Total"}, inplace = True)
    
    total_elec = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_elec)) : 
            total_elec[j] += df_elec.iloc[i,j]
    df_elec.loc[len(df_elec)] = total_elec
    df_elec.rename(index={len(df_elec)-1 : "Total"}, inplace = True)
    
    total_HT = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_HT)) : 
            total_HT[j] += df_HT.iloc[i,j]
    df_HT.loc[len(df_HT)] = total_HT
    df_HT.rename(index={len(df_HT)-1 : "Total"}, inplace = True)
    
    total_LTdec = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_LTdec)) : 
            total_LTdec[j] += df_LTdec.iloc[i,j]
    df_LTdec.loc[len(df_LTdec)] = total_LTdec
    df_LTdec.rename(index={len(df_LTdec)-1 : "Total"}, inplace = True)
    
    total_LTdhn = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df_LTdhn)) : 
            total_LTdhn[j] += df_LTdhn.iloc[i,j]
    df_LTdhn.loc[len(df_LTdhn)] = total_LTdhn
    df_LTdhn.rename(index={len(df_LTdhn)-1 : "Total"}, inplace = True)
    
    dicofdf_pass[cases[c]] = df_pass
    dicofdf_freight[cases[c]] = df_freight
    dicofdf_elec[cases[c]] = df_elec
    dicofdf_HT[cases[c]] = df_HT
    dicofdf_LTdhn[cases[c]] = df_LTdhn
    dicofdf_LTdec[cases[c]] = df_LTdec

#%%
dicofdf_perc_freight = {}
dicofdf_perc_pass = {}
dicofdf_perc_elec = {}
dicofdf_perc_HT = {}
dicofdf_perc_LTdec = {}
dicofdf_perc_LTdhn = {}

for c in range(len(cases)) : 
    df_freight = dicofdf_freight[cases[c]]
    df_perc_freight = pd.DataFrame(columns = years)
    df_pass = dicofdf_pass[cases[c]]
    df_perc_pass = pd.DataFrame(columns = years)
    df_elec = dicofdf_elec[cases[c]]
    df_perc_elec = pd.DataFrame(columns = years)
    df_HT = dicofdf_HT[cases[c]]
    df_perc_HT = pd.DataFrame(columns = years)
    df_LTdec = dicofdf_LTdec[cases[c]]
    df_perc_LTdec = pd.DataFrame(columns = years)
    df_LTdhn = dicofdf_LTdhn[cases[c]]
    df_perc_LTdhn = pd.DataFrame(columns = years)
    
    for i in range(len(df_freight)-1) : 
        perc_freight = np.zeros(7)
        for y in range(len(years)) : 
            perc_freight[y] = df_freight.iloc[i,y]/df_freight.iloc[-1,y]
        df_perc_freight.loc[len(df_perc_freight)] = perc_freight
        df_perc_freight.rename(index={len(df_perc_freight)-1 : df_freight.index[i]}, inplace = True)
        
    for i in range(len(df_pass)-1) : 
        perc_pass = np.zeros(7)
        for y in range(len(years)) : 
            perc_pass[y] = df_pass.iloc[i,y]/df_pass.iloc[-1,y]
        df_perc_pass.loc[len(df_perc_pass)] = perc_pass
        df_perc_pass.rename(index={len(df_perc_pass)-1 : df_pass.index[i]}, inplace = True)
    
    for i in range(len(df_elec)-1) : 
        perc_elec = np.zeros(7)
        for y in range(len(years)) : 
            perc_elec[y] = df_elec.iloc[i,y]/df_elec.iloc[-1,y]
        df_perc_elec.loc[len(df_perc_elec)] = perc_elec
        df_perc_elec.rename(index={len(df_perc_elec)-1 : df_elec.index[i]}, inplace = True)
        
    for i in range(len(df_HT)-1) : 
        perc_HT = np.zeros(7)
        for y in range(len(years)) : 
            perc_HT[y] = df_HT.iloc[i,y]/df_HT.iloc[-1,y]
        df_perc_HT.loc[len(df_perc_HT)] = perc_HT
        df_perc_HT.rename(index={len(df_perc_HT)-1 : df_HT.index[i]}, inplace = True)
        
    for i in range(len(df_LTdec)-1) : 
        perc_LTdec = np.zeros(7)
        for y in range(len(years)) : 
            perc_LTdec[y] = df_LTdec.iloc[i,y]/df_LTdec.iloc[-1,y]
        df_perc_LTdec.loc[len(df_perc_LTdec)] = perc_LTdec
        df_perc_LTdec.rename(index={len(df_perc_LTdec)-1 : df_LTdec.index[i]}, inplace = True)

    for i in range(len(df_freight)-1) : 
        perc_LTdhn = np.zeros(7)
        for y in range(len(years)) : 
            perc_LTdhn[y] = df_LTdhn.iloc[i,y]/df_LTdhn.iloc[-1,y]
        df_perc_LTdhn.loc[len(df_perc_LTdhn)] = perc_LTdhn
        df_perc_LTdhn.rename(index={len(df_perc_LTdhn)-1 : df_LTdhn.index[i]}, inplace = True)
        
    dicofdf_perc_freight[cases[c]] = df_perc_freight
    dicofdf_perc_pass[cases[c]] = df_perc_pass
    dicofdf_perc_elec[cases[c]] = df_perc_elec
    dicofdf_perc_HT[cases[c]] = df_perc_HT
    dicofdf_perc_LTdec[cases[c]] = df_perc_LTdec
    dicofdf_perc_LTdhn[cases[c]] = df_perc_LTdhn


changes = pd.DataFrame(columns = years)


#%% FREIGHT
colors = {" TRAIN_FREIGHT " : "royalblue", " BOAT_FREIGHT_DIESEL " : "dimgrey", " BOAT_FREIGHT_NG " : "darkorange", " BOAT_FREIGHT_METHANOL " : "fuchsia", " TRUCK_DIESEL " : "darkgrey", " TRUCK_FUEL_CELL " : "violet", " TRUCK_ELEC " : "dodgerblue", " TRUCK_NG " : "moccasin", " TRUCK_METHANOL " : "orchid"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_freight["nom"])) : 
    tech = dicofdf_perc_freight["nom"].index[i]
    if tech not in dicofdf_perc_freight["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["low"].loc[len(dicofdf_perc_freight["low"])] = toadd
        dicofdf_perc_freight["low"].rename(index = {len(dicofdf_perc_freight["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_freight["nom"])) : 
    tech = dicofdf_perc_freight["nom"].index[i]
    if tech not in dicofdf_perc_freight["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["high"].loc[len(dicofdf_perc_freight["high"])] = toadd
        dicofdf_perc_freight["high"].rename(index = {len(dicofdf_perc_freight["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_freight["low"])) : 
    tech = dicofdf_perc_freight["low"].index[i]
    if tech not in dicofdf_perc_freight["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["nom"].loc[len(dicofdf_perc_freight["nom"])] = toadd
        dicofdf_perc_freight["nom"].rename(index = {len(dicofdf_perc_freight["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_freight["low"])) : 
    tech = dicofdf_perc_freight["low"].index[i]
    if tech not in dicofdf_perc_freight["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["high"].loc[len(dicofdf_perc_freight["high"])] = toadd
        dicofdf_perc_freight["high"].rename(index = {len(dicofdf_perc_freight["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_freight["high"])) : 
    tech = dicofdf_perc_freight["high"].index[i]
    if tech not in dicofdf_perc_freight["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["nom"].loc[len(dicofdf_perc_freight["nom"])] = toadd
        dicofdf_perc_freight["nom"].rename(index = {len(dicofdf_perc_freight["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_freight["high"])) : 
    tech = dicofdf_perc_freight["high"].index[i]
    if tech not in dicofdf_perc_freight["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_freight["low"].loc[len(dicofdf_perc_freight["low"])] = toadd
        dicofdf_perc_freight["low"].rename(index = {len(dicofdf_perc_freight["low"])-1 : i}, inplace = True)        
        

for i in range(len(dicofdf_perc_freight["nom"])) : 
    tech = dicofdf_perc_freight["nom"].index[i]
    plt.scatter(0, dicofdf_perc_freight["nom"].iloc[i,2], color=colors[tech], label = tech, s=100)
for i in range(len(dicofdf_perc_freight["low"])) : 
    tech = dicofdf_perc_freight["low"].index[i]
    plt.scatter(1, dicofdf_perc_freight["low"].iloc[i,2], color=colors[tech], s = 100)
for i in range(len(dicofdf_perc_freight["high"])) : 
    tech = dicofdf_perc_freight["high"].index[i]
    plt.scatter(2, dicofdf_perc_freight["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_freight_2030_low = 0 
for i in range(len(dicofdf_perc_freight["nom"])):
    change_freight_2030_low += np.abs(dicofdf_perc_freight["nom"].iloc[i,2]-dicofdf_perc_freight["low"].iloc[i,2]) 

change_freight_2030_high = 0 
for i in range(len(dicofdf_perc_freight["nom"])):
    change_freight_2030_high += np.abs(dicofdf_perc_freight["nom"].iloc[i,2]-dicofdf_perc_freight["high"].iloc[i,2]) 

change_freight_2050_low = 0 
for i in range(len(dicofdf_perc_freight["nom"])):
    change_freight_2050_low += np.abs(dicofdf_perc_freight["nom"].iloc[i,-1]-dicofdf_perc_freight["low"].iloc[i,-1]) 

change_freight_2050_high = 0 
for i in range(len(dicofdf_perc_freight["nom"])):
    change_freight_2050_high += np.abs(dicofdf_perc_freight["nom"].iloc[i,-1]-dicofdf_perc_freight["high"].iloc[i,-1]) 

#%%
changes_freight = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_freight["nom"])) : 
        tech = dicofdf_perc_freight["nom"].index[i]
        row_low = dicofdf_perc_freight["low"].loc[tech]
        row_high = dicofdf_perc_freight["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_freight["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_freight["nom"].iloc[i,y]-row_high[years[y]])
    changes_freight["low"].append(c_low)
    changes_freight["high"].append(c_high)
"""
fig = plt.figure(figsize=(12, 7))
plt.plot(years, changes_freight["low"], label = "low", color = 'cornflowerblue', linestyle = 'dotted')
plt.plot(years, changes_freight["high"], label = "high", color = 'midnightblue', linestyle = 'dotted')
plt.scatter(years, changes_freight["low"], color = 'cornflowerblue')#, marker = '^')
plt.scatter(years, changes_freight["high"], color = 'midnightblue')
sns.despine()
plt.legend()
fig.savefig(folder_low + 'graph_freight.svg')
plt.show()"""
#%% new

changes_perc_low_freight = pd.DataFrame(columns = years)
changes_perc_high_freight = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_freight["nom"])):
    tech = dicofdf_perc_freight["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_freight["nom"].iloc[i,y]+dicofdf_perc_freight["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_freight["nom"].iloc[i,y]+dicofdf_perc_freight["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_freight.loc[len(changes_perc_low_freight)] = toadd_low
        changes_perc_low_freight.rename(index={len(changes_perc_low_freight)-1 : tech}, inplace = True)
        changes_perc_high_freight.loc[len(changes_perc_high_freight)] = toadd_high
        changes_perc_high_freight.rename(index={len(changes_perc_high_freight)-1 : tech}, inplace = True)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_freight)): 
    tech = changes_perc_low_freight.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_freight.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_freight_low.svg', bbox_inches = 'tight')
plt.show()
#%% new 2
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_freight)): 
    tech = changes_perc_high_freight.index[i]
    plt.plot(years, [changes_perc_high_freight.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
#plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.05)
fig.savefig(folder_low + 'tech_changes_freight_high.svg', bbox_inches = 'tight')
plt.show()

#%% PASSENGER
colors = {" TRAMWAY_TROLLEY " : "dodgerblue", " BUS_COACH_DIESEL " : "dimgrey", " BUS_COACH_HYDIESEL " : "gray", " BUS_COACH_CNG_STOICH " : "orange", " BUS_COACH_FC_HYBRIDH2 " : "violet", " TRAIN_PUB " : "blue", " CAR_GASOLINE " : "black", " CAR_DIESEL " : "lightgray", " CAR_NG " : "moccasin", " CAR_METHANOL ":"orchid", " CAR_HEV " : "salmon", " CAR_PHEV " : "lightsalmon", " CAR_BEV " : "deepskyblue", " CAR_FUEL_CELL " : "magenta"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_pass["nom"])) : 
    tech = dicofdf_perc_pass["nom"].index[i]
    if tech not in dicofdf_perc_pass["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["low"].loc[len(dicofdf_perc_pass["low"])] = toadd
        dicofdf_perc_pass["low"].rename(index = {len(dicofdf_perc_pass["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_pass["nom"])) : 
    tech = dicofdf_perc_pass["nom"].index[i]
    if tech not in dicofdf_perc_pass["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["high"].loc[len(dicofdf_perc_pass["high"])] = toadd
        dicofdf_perc_pass["high"].rename(index = {len(dicofdf_perc_pass["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_pass["low"])) : 
    tech = dicofdf_perc_pass["low"].index[i]
    if tech not in dicofdf_perc_pass["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["nom"].loc[len(dicofdf_perc_pass["nom"])] = toadd
        dicofdf_perc_pass["nom"].rename(index = {len(dicofdf_perc_pass["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_pass["low"])) : 
    tech = dicofdf_perc_pass["low"].index[i]
    if tech not in dicofdf_perc_pass["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["high"].loc[len(dicofdf_perc_pass["high"])] = toadd
        dicofdf_perc_pass["high"].rename(index = {len(dicofdf_perc_pass["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_pass["high"])) : 
    tech = dicofdf_perc_pass["high"].index[i]
    if tech not in dicofdf_perc_pass["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["nom"].loc[len(dicofdf_perc_pass["nom"])] = toadd
        dicofdf_perc_pass["nom"].rename(index = {len(dicofdf_perc_pass["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_pass["high"])) : 
    tech = dicofdf_perc_pass["high"].index[i]
    if tech not in dicofdf_perc_pass["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_pass["low"].loc[len(dicofdf_perc_pass["low"])] = toadd
        dicofdf_perc_pass["low"].rename(index = {len(dicofdf_perc_pass["low"])-1 : i}, inplace = True) 



for i in range(len(dicofdf_perc_pass["nom"])) : 
    tech = dicofdf_perc_pass["nom"].index[i]
    plt.scatter(0, dicofdf_perc_pass["low"].iloc[i,2], color=colors[tech], label = tech, s=100)
    plt.scatter(1, dicofdf_perc_pass["nom"].iloc[i,2], color=colors[tech], s = 100)
    plt.scatter(2, dicofdf_perc_pass["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_pass_2030_low = 0 
for i in range(len(dicofdf_perc_pass["nom"])):
    change_pass_2030_low += np.abs(dicofdf_perc_pass["nom"].iloc[i,2]-dicofdf_perc_pass["low"].iloc[i,2]) 

change_pass_2030_high = 0 
for i in range(len(dicofdf_perc_pass["nom"])):
    change_pass_2030_high += np.abs(dicofdf_perc_pass["nom"].iloc[i,2]-dicofdf_perc_pass["high"].iloc[i,2]) 

change_pass_2050_low = 0 
for i in range(len(dicofdf_perc_pass["nom"])):
    change_pass_2050_low += np.abs(dicofdf_perc_pass["nom"].iloc[i,-1]-dicofdf_perc_pass["low"].iloc[i,-1]) 

change_pass_2050_high = 0 
for i in range(len(dicofdf_perc_pass["nom"])):
    change_pass_2050_high += np.abs(dicofdf_perc_pass["nom"].iloc[i,-1]-dicofdf_perc_pass["high"].iloc[i,-1]) 
#%%
changes_pass = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_pass["nom"])) : 
        tech = dicofdf_perc_pass["nom"].index[i]
        row_low = dicofdf_perc_pass["low"].loc[tech]
        row_high = dicofdf_perc_pass["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_pass["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_pass["nom"].iloc[i,y]-row_high[years[y]])
    changes_pass["low"].append(c_low)
    changes_pass["high"].append(c_high)
"""
fig = plt.figure(figsize=(12, 7))
plt.plot(years, changes_pass["low"], color = "lightskyblue", linestyle = 'dotted', label = "low")
plt.plot(years, changes_pass["high"], color = 'steelblue', linestyle = 'dotted', label = "high")
plt.scatter(years, changes_pass["low"], color = 'lightskyblue')
plt.scatter(years, changes_pass["high"], color = 'steelblue')
sns.despine()
plt.legend()
fig.savefig(folder_low + 'graph_pass.svg')
plt.show()"""

#%% new

changes_perc_low_pass = pd.DataFrame(columns = years)
changes_perc_high_pass = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_pass["nom"])):
    tech = dicofdf_perc_pass["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_pass["nom"].iloc[i,y]+dicofdf_perc_pass["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_pass["nom"].iloc[i,y]+dicofdf_perc_pass["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_pass.loc[len(changes_perc_low_pass)] = toadd_low
        changes_perc_low_pass.rename(index={len(changes_perc_low_pass)-1 : tech}, inplace = True)
        changes_perc_high_pass.loc[len(changes_perc_high_pass)] = toadd_high
        changes_perc_high_pass.rename(index={len(changes_perc_high_pass)-1 : tech}, inplace = True)
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_pass)): 
    tech = changes_perc_low_pass.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_pass.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_pass_low.svg', bbox_inches = 'tight')
plt.show()

#%% new 2
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_pass)): 
    tech = changes_perc_high_pass.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_high_pass.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_pass_high.svg', bbox_inches = 'tight')
plt.show()



#%% ELEC

colors = {"OTHERS" : "lightgrey"," IND_COGEN_WASTE " : "olive", " IND_COGEN_GAS ": "orange", " NUCLEAR ":"deeppink", " CCGT ":"darkorange", " CCGT_AMMONIA ":"slateblue", " COAL_US " : "black", " COAL_IGCC " : "dimgray", " PV " : "yellow", " WIND_ONSHORE " : "lawngreen", " WIND_OFFSHORE " : "green", " HYDRO_RIVER " : "blue", " GEOTHERMAL " : "firebrick", " ELECTRICITY " : "dodgerblue"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_elec["nom"])) : 
    tech = dicofdf_perc_elec["nom"].index[i]
    if tech not in dicofdf_perc_elec["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["low"].loc[len(dicofdf_perc_elec["low"])] = toadd
        dicofdf_perc_elec["low"].rename(index = {len(dicofdf_perc_elec["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_elec["nom"])) : 
    tech = dicofdf_perc_elec["nom"].index[i]
    if tech not in dicofdf_perc_elec["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["high"].loc[len(dicofdf_perc_elec["high"])] = toadd
        dicofdf_perc_elec["high"].rename(index = {len(dicofdf_perc_elec["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_elec["low"])) : 
    tech = dicofdf_perc_elec["low"].index[i]
    if tech not in dicofdf_perc_elec["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["nom"].loc[len(dicofdf_perc_elec["nom"])] = toadd
        dicofdf_perc_elec["nom"].rename(index = {len(dicofdf_perc_elec["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_elec["low"])) : 
    tech = dicofdf_perc_elec["low"].index[i]
    if tech not in dicofdf_perc_elec["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["high"].loc[len(dicofdf_perc_elec["high"])] = toadd
        dicofdf_perc_elec["high"].rename(index = {len(dicofdf_perc_elec["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_elec["high"])) : 
    tech = dicofdf_perc_elec["high"].index[i]
    if tech not in dicofdf_perc_elec["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["nom"].loc[len(dicofdf_perc_elec["nom"])] = toadd
        dicofdf_perc_elec["nom"].rename(index = {len(dicofdf_perc_elec["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_elec["high"])) : 
    tech = dicofdf_perc_elec["high"].index[i]
    if tech not in dicofdf_perc_elec["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_elec["low"].loc[len(dicofdf_perc_elec["low"])] = toadd
        dicofdf_perc_elec["low"].rename(index = {len(dicofdf_perc_elec["low"])-1 : i}, inplace = True) 



for i in range(len(dicofdf_perc_elec["nom"])) : 
    tech = dicofdf_perc_elec["nom"].index[i]
    plt.scatter(0, dicofdf_perc_elec["low"].iloc[i,2], color=colors[tech], label = tech, s = 100)
    plt.scatter(1, dicofdf_perc_elec["nom"].iloc[i,2], color=colors[tech], s = 100)
    plt.scatter(2, dicofdf_perc_elec["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_elec_2030_low = 0 
for i in range(len(dicofdf_perc_elec["nom"])):
    change_elec_2030_low += np.abs(dicofdf_perc_elec["nom"].iloc[i,2]-dicofdf_perc_elec["low"].iloc[i,2]) 

change_elec_2030_high = 0 
for i in range(len(dicofdf_perc_elec["nom"])):
    change_elec_2030_high += np.abs(dicofdf_perc_elec["nom"].iloc[i,2]-dicofdf_perc_elec["high"].iloc[i,2]) 

change_elec_2050_low = 0 
for i in range(len(dicofdf_perc_elec["nom"])):
    change_elec_2050_low += np.abs(dicofdf_perc_elec["nom"].iloc[i,-1]-dicofdf_perc_elec["low"].iloc[i,-1]) 

change_elec_2050_high = 0 
for i in range(len(dicofdf_perc_elec["nom"])):
    change_elec_2050_high += np.abs(dicofdf_perc_elec["nom"].iloc[i,-1]-dicofdf_perc_elec["high"].iloc[i,-1]) 

#%%
changes_elec = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_elec["nom"])) : 
        tech = dicofdf_perc_elec["nom"].index[i]
        row_low = dicofdf_perc_elec["low"].loc[tech]
        row_high = dicofdf_perc_elec["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_elec["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_elec["nom"].iloc[i,y]-row_high[years[y]])
    changes_elec["low"].append(c_low)
    changes_elec["high"].append(c_high)

#%% new

changes_perc_low_elec = pd.DataFrame(columns = years)
changes_perc_high_elec = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_elec["nom"])):
    tech = dicofdf_perc_elec["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_elec["nom"].iloc[i,y]+dicofdf_perc_elec["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_elec["nom"].iloc[i,y]+dicofdf_perc_elec["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_elec.loc[len(changes_perc_low_elec)] = toadd_low
        changes_perc_low_elec.rename(index={len(changes_perc_low_elec)-1 : tech}, inplace = True)
        changes_perc_high_elec.loc[len(changes_perc_high_elec)] = toadd_high
        changes_perc_high_elec.rename(index={len(changes_perc_high_elec)-1 : tech}, inplace = True)
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_elec)): 
    tech = changes_perc_low_elec.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_elec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_elec_low.svg', bbox_inches = 'tight')
plt.show()


#%% new 2

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_elec)): 
    tech = changes_perc_high_elec.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_high_elec.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_elec_high.svg', bbox_inches = 'tight')
plt.show()


#%% HT

colors = {" IND_COGEN_GAS ":"orange", " IND_COGEN_WOOD ":"peru", " IND_COGEN_WASTE " : "olive", " IND_BOILER_GAS " : "moccasin", " IND_BOILER_WOOD " : "goldenrod", " IND_BOILER_OIL " : "blueviolet", " IND_BOILER_COAL " : "black", " IND_BOILER_WASTE " : "olivedrab", " IND_DIRECT_ELEC " : "royalblue"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_HT["nom"])) : 
    tech = dicofdf_perc_HT["nom"].index[i]
    if tech not in dicofdf_perc_HT["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["low"].loc[len(dicofdf_perc_HT["low"])] = toadd
        dicofdf_perc_HT["low"].rename(index = {len(dicofdf_perc_HT["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_HT["nom"])) : 
    tech = dicofdf_perc_HT["nom"].index[i]
    if tech not in dicofdf_perc_HT["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["high"].loc[len(dicofdf_perc_HT["high"])] = toadd
        dicofdf_perc_HT["high"].rename(index = {len(dicofdf_perc_HT["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_HT["low"])) : 
    tech = dicofdf_perc_HT["low"].index[i]
    if tech not in dicofdf_perc_HT["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["nom"].loc[len(dicofdf_perc_HT["nom"])] = toadd
        dicofdf_perc_HT["nom"].rename(index = {len(dicofdf_perc_HT["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_HT["low"])) : 
    tech = dicofdf_perc_HT["low"].index[i]
    if tech not in dicofdf_perc_HT["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["high"].loc[len(dicofdf_perc_HT["high"])] = toadd
        dicofdf_perc_HT["high"].rename(index = {len(dicofdf_perc_HT["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_HT["high"])) : 
    tech = dicofdf_perc_HT["high"].index[i]
    if tech not in dicofdf_perc_HT["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["nom"].loc[len(dicofdf_perc_HT["nom"])] = toadd
        dicofdf_perc_HT["nom"].rename(index = {len(dicofdf_perc_HT["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_HT["high"])) : 
    tech = dicofdf_perc_HT["high"].index[i]
    if tech not in dicofdf_perc_HT["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_HT["low"].loc[len(dicofdf_perc_HT["low"])] = toadd
        dicofdf_perc_HT["low"].rename(index = {len(dicofdf_perc_HT["low"])-1 : i}, inplace = True)        


for i in range(len(dicofdf_perc_HT["nom"])) : 
    tech = dicofdf_perc_HT["nom"].index[i]
    plt.scatter(0, dicofdf_perc_HT["low"].iloc[i,2], color=colors[tech], label = tech, s = 100)
    plt.scatter(1, dicofdf_perc_HT["nom"].iloc[i,2], color=colors[tech], s = 100)
    plt.scatter(2, dicofdf_perc_HT["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_HT_2030_low = 0 
for i in range(len(dicofdf_perc_HT["nom"])):
    change_HT_2030_low += np.abs(dicofdf_perc_HT["nom"].iloc[i,2]-dicofdf_perc_HT["low"].iloc[i,2]) 

change_HT_2030_high = 0 
for i in range(len(dicofdf_perc_HT["nom"])):
    change_HT_2030_high += np.abs(dicofdf_perc_HT["nom"].iloc[i,2]-dicofdf_perc_HT["high"].iloc[i,2]) 

change_HT_2050_low = 0 
for i in range(len(dicofdf_perc_HT["nom"])):
    change_HT_2050_low += np.abs(dicofdf_perc_HT["nom"].iloc[i,-1]-dicofdf_perc_HT["low"].iloc[i,-1]) 

change_HT_2050_high = 0 
for i in range(len(dicofdf_perc_HT["nom"])):
    change_HT_2050_high += np.abs(dicofdf_perc_HT["nom"].iloc[i,-1]-dicofdf_perc_HT["high"].iloc[i,-1]) 
#%%
changes_HT = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_HT["nom"])) : 
        tech = dicofdf_perc_HT["nom"].index[i]
        row_low = dicofdf_perc_HT["low"].loc[tech]
        row_high = dicofdf_perc_HT["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_HT["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_HT["nom"].iloc[i,y]-row_high[years[y]])
    changes_HT["low"].append(c_low)
    changes_HT["high"].append(c_high)

#%% new

changes_perc_low_HT = pd.DataFrame(columns = years)
changes_perc_high_HT = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_HT["nom"])):
    tech = dicofdf_perc_HT["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_HT["nom"].iloc[i,y]+dicofdf_perc_HT["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_HT["nom"].iloc[i,y]+dicofdf_perc_HT["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_HT.loc[len(changes_perc_low_HT)] = toadd_low
        changes_perc_low_HT.rename(index={len(changes_perc_low_HT)-1 : tech}, inplace = True)
        changes_perc_high_HT.loc[len(changes_perc_high_HT)] = toadd_high
        changes_perc_high_HT.rename(index={len(changes_perc_high_HT)-1 : tech}, inplace = True)
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_HT)): 
    tech = changes_perc_low_HT.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_HT.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_HT_low.svg', bbox_inches = 'tight')
plt.show()

#%% new 2

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_HT)): 
    tech = changes_perc_high_HT.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_high_HT.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_HT_high.svg', bbox_inches = 'tight')
plt.show()


#%% LTdec

colors = {" DHN_HP_ELEC " : "blue", " DHN_COGEN_GAS " : "orange", " DHN_COGEN_WOOD " : "sandybrown", " DHN_COGEN_WASTE " : "olive", " DHN_COGEN_WET_BIOMASS " : "seagreen", " DHN_COGEN_BIO_HYDROLYSIS " : "springgreen", " DHN_BOILER_GAS " : "darkorange", " DHN_BOILER_WOOD " : "sienna", " DHN_BOILER_OIL " : "blueviolet", " DHN_DEEP_GEO " : "firebrick", " DHN_SOLAR " : "gold", " DEC_HP_ELEC " : "cornflowerblue", " DEC_THHP_GAS " : "lightsalmon", " DEC_COGEN_GAS " : "goldenrod", " DEC_COGEN_OIL " : "mediumpurple", " DEC_ADVCOGEN_GAS " : "burlywood", " DEC_ADVCOGEN_H2 " : "violet", " DEC_BOILER_GAS " : "moccasin", " DEC_BOILER_WOOD " : "peru", " DEC_BOILER_OIL " : "darkorchid", " DEC_SOLAR " : "yellow", " DEC_DIRECT_ELEC " : "deepskyblue"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_LTdec["nom"])) : 
    tech = dicofdf_perc_LTdec["nom"].index[i]
    if tech not in dicofdf_perc_LTdec["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["low"].loc[len(dicofdf_perc_LTdec["low"])] = toadd
        dicofdf_perc_LTdec["low"].rename(index = {len(dicofdf_perc_LTdec["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_LTdec["nom"])) : 
    tech = dicofdf_perc_LTdec["nom"].index[i]
    if tech not in dicofdf_perc_LTdec["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["high"].loc[len(dicofdf_perc_LTdec["high"])] = toadd
        dicofdf_perc_LTdec["high"].rename(index = {len(dicofdf_perc_LTdec["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_LTdec["low"])) : 
    tech = dicofdf_perc_LTdec["low"].index[i]
    if tech not in dicofdf_perc_LTdec["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["nom"].loc[len(dicofdf_perc_LTdec["nom"])] = toadd
        dicofdf_perc_LTdec["nom"].rename(index = {len(dicofdf_perc_LTdec["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_LTdec["low"])) : 
    tech = dicofdf_perc_LTdec["low"].index[i]
    if tech not in dicofdf_perc_LTdec["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["high"].loc[len(dicofdf_perc_LTdec["high"])] = toadd
        dicofdf_perc_LTdec["high"].rename(index = {len(dicofdf_perc_LTdec["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_LTdec["high"])) : 
    tech = dicofdf_perc_LTdec["high"].index[i]
    if tech not in dicofdf_perc_LTdec["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["nom"].loc[len(dicofdf_perc_LTdec["nom"])] = toadd
        dicofdf_perc_LTdec["nom"].rename(index = {len(dicofdf_perc_LTdec["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_LTdec["high"])) : 
    tech = dicofdf_perc_LTdec["high"].index[i]
    if tech not in dicofdf_perc_LTdec["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdec["low"].loc[len(dicofdf_perc_LTdec["low"])] = toadd
        dicofdf_perc_LTdec["low"].rename(index = {len(dicofdf_perc_LTdec["low"])-1 : i}, inplace = True)        


for i in range(len(dicofdf_perc_LTdec["nom"])) : 
    tech = dicofdf_perc_LTdec["nom"].index[i]
    plt.scatter(0, dicofdf_perc_LTdec["low"].iloc[i,2], color=colors[tech], label = tech, s = 100)
    plt.scatter(1, dicofdf_perc_LTdec["nom"].iloc[i,2], color=colors[tech], s = 100)
    plt.scatter(2, dicofdf_perc_LTdec["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_LTdec_2030_low = 0 
for i in range(len(dicofdf_perc_LTdec["nom"])):
    change_LTdec_2030_low += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,2]-dicofdf_perc_LTdec["low"].iloc[i,2]) 

change_LTdec_2030_high = 0 
for i in range(len(dicofdf_perc_LTdec["nom"])):
    change_LTdec_2030_high += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,2]-dicofdf_perc_LTdec["high"].iloc[i,2]) 

change_LTdec_2050_low = 0 
for i in range(len(dicofdf_perc_LTdec["nom"])):
    change_LTdec_2050_low += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,-1]-dicofdf_perc_LTdec["low"].iloc[i,-1]) 

change_LTdec_2050_high = 0 
for i in range(len(dicofdf_perc_LTdec["nom"])):
    change_LTdec_2050_high += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,-1]-dicofdf_perc_LTdec["high"].iloc[i,-1]) 

#%%
changes_LTdec = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_LTdec["nom"])) : 
        tech = dicofdf_perc_LTdec["nom"].index[i]
        row_low = dicofdf_perc_LTdec["low"].loc[tech]
        row_high = dicofdf_perc_LTdec["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_LTdec["nom"].iloc[i,y]-row_high[years[y]])
    changes_LTdec["low"].append(c_low)
    changes_LTdec["high"].append(c_high)

#%% new
changes_perc_low_LTdec = pd.DataFrame(columns = years)
changes_perc_high_LTdec = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_LTdec["nom"])):
    tech = dicofdf_perc_LTdec["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_LTdec["nom"].iloc[i,y]+dicofdf_perc_LTdec["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_LTdec["nom"].iloc[i,y]+dicofdf_perc_LTdec["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_LTdec.loc[len(changes_perc_low_LTdec)] = toadd_low
        changes_perc_low_LTdec.rename(index={len(changes_perc_low_LTdec)-1 : tech}, inplace = True)
        changes_perc_high_LTdec.loc[len(changes_perc_high_LTdec)] = toadd_high
        changes_perc_high_LTdec.rename(index={len(changes_perc_high_LTdec)-1 : tech}, inplace = True)

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_LTdec)): 
    tech = changes_perc_low_LTdec.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_LTdec_low.svg', bbox_inches = 'tight')
plt.show()


#%% new 2

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_LTdec)): 
    tech = changes_perc_high_LTdec.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_LTdec_high.svg', bbox_inches = 'tight')
plt.show()


#%% LTdhn

colors = {" DHN_HP_ELEC " : "blue", " DHN_COGEN_GAS " : "orange", " DHN_COGEN_WOOD " : "sandybrown", " DHN_COGEN_WASTE " : "olive", " DHN_COGEN_WET_BIOMASS " : "seagreen", " DHN_COGEN_BIO_HYDROLYSIS " : "springgreen", " DHN_BOILER_GAS " : "darkorange", " DHN_BOILER_WOOD " : "sienna", " DHN_BOILER_OIL " : "blueviolet", " DHN_DEEP_GEO " : "firebrick", " DHN_SOLAR " : "gold", " DEC_HP_ELEC " : "cornflowerblue", " DEC_THHP_GAS " : "lightsalmon", " DEC_COGEN_GAS " : "goldenrod", " DEC_COGEN_OIL " : "mediumpurple", " DEC_ADVCOGEN_GAS " : "burlywood", " DEC_ADVCOGEN_H2 " : "violet", " DEC_BOILER_GAS " : "moccasin", " DEC_BOILER_WOOD " : "peru", " DEC_BOILER_OIL " : "darkorchid", " DEC_SOLAR " : "yellow", " DEC_DIRECT_ELEC " : "deepskyblue"}

idxtoadd = []
toadd = np.zeros(7)
for i in range(len(dicofdf_perc_LTdhn["nom"])) : 
    tech = dicofdf_perc_LTdhn["nom"].index[i]
    if tech not in dicofdf_perc_LTdhn["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["low"].loc[len(dicofdf_perc_LTdhn["low"])] = toadd
        dicofdf_perc_LTdhn["low"].rename(index = {len(dicofdf_perc_LTdhn["low"])-1 : i}, inplace = True)
        
idxtoadd = []
for i in range(len(dicofdf_perc_LTdhn["nom"])) : 
    tech = dicofdf_perc_LTdhn["nom"].index[i]
    if tech not in dicofdf_perc_LTdhn["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["high"].loc[len(dicofdf_perc_LTdhn["high"])] = toadd
        dicofdf_perc_LTdhn["high"].rename(index = {len(dicofdf_perc_LTdhn["high"])-1 : i}, inplace = True)   

idxtoadd = []
for i in range(len(dicofdf_perc_LTdhn["low"])) : 
    tech = dicofdf_perc_LTdhn["low"].index[i]
    if tech not in dicofdf_perc_LTdhn["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["nom"].loc[len(dicofdf_perc_LTdhn["nom"])] = toadd
        dicofdf_perc_LTdhn["nom"].rename(index = {len(dicofdf_perc_LTdhn["nom"])-1 : i}, inplace = True) 
    
idxtoadd = []
for i in range(len(dicofdf_perc_LTdhn["low"])) : 
    tech = dicofdf_perc_LTdhn["low"].index[i]
    if tech not in dicofdf_perc_LTdhn["high"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["high"].loc[len(dicofdf_perc_LTdhn["high"])] = toadd
        dicofdf_perc_LTdhn["high"].rename(index = {len(dicofdf_perc_LTdhn["high"])-1 : i}, inplace = True) 

idxtoadd = []
for i in range(len(dicofdf_perc_LTdhn["high"])) : 
    tech = dicofdf_perc_LTdhn["high"].index[i]
    if tech not in dicofdf_perc_LTdhn["nom"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["nom"].loc[len(dicofdf_perc_LTdhn["nom"])] = toadd
        dicofdf_perc_LTdhn["nom"].rename(index = {len(dicofdf_perc_LTdhn["nom"])-1 : i}, inplace = True)

idxtoadd = []
for i in range(len(dicofdf_perc_LTdhn["high"])) : 
    tech = dicofdf_perc_LTdhn["high"].index[i]
    if tech not in dicofdf_perc_LTdhn["low"].index : 
        idxtoadd.append(tech)
if idxtoadd != [] : 
    for i in idxtoadd : 
        dicofdf_perc_LTdhn["low"].loc[len(dicofdf_perc_LTdhn["low"])] = toadd
        dicofdf_perc_LTdhn["low"].rename(index = {len(dicofdf_perc_LTdhn["low"])-1 : i}, inplace = True)        

for i in range(len(dicofdf_perc_LTdhn["nom"])) : 
    tech = dicofdf_perc_LTdhn["nom"].index[i]
    plt.scatter(0, dicofdf_perc_LTdhn["low"].iloc[i,2], color=colors[tech], label = tech, s = 100)
    plt.scatter(1, dicofdf_perc_LTdhn["nom"].iloc[i,2], color=colors[tech], s = 100)
    plt.scatter(2, dicofdf_perc_LTdhn["high"].iloc[i,2], color=colors[tech], s = 100)
#plt.legend()
sns.despine()
plt.show()

change_LTdhn_2030_low = 0 
for i in range(len(dicofdf_perc_LTdhn["nom"])):
    change_LTdhn_2030_low += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,2]-dicofdf_perc_LTdhn["low"].iloc[i,2]) 

change_LTdhn_2030_high = 0 
for i in range(len(dicofdf_perc_LTdhn["nom"])):
    change_LTdhn_2030_high += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,2]-dicofdf_perc_LTdhn["high"].iloc[i,2]) 

change_LTdhn_2050_low = 0 
for i in range(len(dicofdf_perc_LTdhn["nom"])):
    change_LTdhn_2050_low += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,-1]-dicofdf_perc_LTdhn["low"].iloc[i,-1]) 

change_LTdhn_2050_high = 0 
for i in range(len(dicofdf_perc_LTdhn["nom"])):
    change_LTdhn_2050_high += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,-1]-dicofdf_perc_LTdhn["high"].iloc[i,-1]) 
#%%
changes_LTdhn = {"low" : [], "high" : []}
for y in range(len(years)) : 
    c_low = 0
    c_high = 0
    for i in range(len(dicofdf_perc_LTdhn["nom"])) : 
        tech = dicofdf_perc_LTdhn["nom"].index[i]
        row_low = dicofdf_perc_LTdhn["low"].loc[tech]
        row_high = dicofdf_perc_LTdhn["high"].loc[tech]
        c_low += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,y]-row_low[years[y]])
        c_high += np.abs(dicofdf_perc_LTdhn["nom"].iloc[i,y]-row_high[years[y]])
    changes_LTdhn["low"].append(c_low)
    changes_LTdhn["high"].append(c_high)

#%% new

changes_perc_low_LTdhn = pd.DataFrame(columns = years)
changes_perc_high_LTdhn = pd.DataFrame(columns = years)
for i in range(len(dicofdf_perc_LTdhn["nom"])):
    tech = dicofdf_perc_LTdhn["nom"].index[i]
    toadd_low = np.zeros(7)
    toadd_high = np.zeros(7)
    for y in range(len(years)) :
        toadd_low[y] = -dicofdf_perc_LTdhn["nom"].iloc[i,y]+dicofdf_perc_LTdhn["low"].iloc[i,y]
        toadd_high[y] = -dicofdf_perc_LTdhn["nom"].iloc[i,y]+dicofdf_perc_LTdhn["high"].iloc[i,y]
    toadd = False
    for y in range(len(years)) : 
        if np.abs(toadd_low[y]) >= 0.01 or np.abs(toadd_high[y]) >= 0.01 : 
            toadd = True
    if toadd : 
        changes_perc_low_LTdhn.loc[len(changes_perc_low_LTdhn)] = toadd_low
        changes_perc_low_LTdhn.rename(index={len(changes_perc_low_LTdhn)-1 : tech}, inplace = True)
        changes_perc_high_LTdhn.loc[len(changes_perc_high_LTdhn)] = toadd_high
        changes_perc_high_LTdhn.rename(index={len(changes_perc_high_LTdhn)-1 : tech}, inplace = True)
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_low_LTdhn)): 
    tech = changes_perc_low_LTdhn.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_low_LTdhn.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_LTdhn_low.svg', bbox_inches = 'tight')
plt.show()

#%% new 2

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
for i in range(len(changes_perc_high_LTdhn)): 
    tech = changes_perc_high_LTdhn.index[i]
    #plt.plot(years, [changes_perc_high_LTdec.iloc[i,j] for j in range(len(years))],  linestyle = 'dotted', label = tech, color = colors[tech])
    plt.plot(years, [changes_perc_high_LTdhn.iloc[i,j] for j in range(len(years))],  linestyle = 'dashed', label = tech, color = colors[tech])
plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{tech}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'tech_changes_LTdhn_high.svg', bbox_inches = 'tight')
plt.show()


#%% GIGA GRAPH LOW
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
plt.plot(years, changes_LTdhn["low"], label = "low_dhn", color = "firebrick", linestyle = 'dotted')
#plt.plot(years, changes_LTdhn["high"], label = "high_dhn", color = 'palevioletred', linestyle = 'dashed')
plt.scatter(years, changes_LTdhn["low"], color = "firebrick")
#plt.scatter(years, changes_LTdhn["high"], color = 'palevioletred')
plt.plot(years, changes_LTdec["low"], label = "low_dec", color = "#FF9999", linestyle = 'dotted')
#plt.plot(years, changes_LTdec["high"], label = "high_dec", color = 'purple', linestyle = 'dashed')
plt.scatter(years, changes_LTdec["low"], color = "#FF9999")
#plt.scatter(years, changes_LTdec["high"], color = 'purple')
plt.plot(years, changes_HT["low"], label = "low_ht", color = "#FF0000", linestyle = 'dotted')
#plt.plot(years, changes_HT["high"], label = "high_ht", color = 'darkred', linestyle = 'dashed')
plt.scatter(years, changes_HT["low"], color = "#FF0000")
#plt.scatter(years, changes_HT["high"], color = 'darkred')
plt.plot(years, changes_elec["low"], label = "low_elec", color = "#00B0F0", linestyle = 'dotted')
#plt.plot(years, changes_elec["high"], label = "high_elec", color = 'darkorange', linestyle = 'dashed')
plt.scatter(years, changes_elec["low"], color = "#00B0F0")
#plt.scatter(years, changes_elec["high"], color = 'darkorange')
plt.plot(years, changes_pass["low"], color = "#FFC000", linestyle = 'dotted', label = "low_pass")
#plt.plot(years, changes_pass["high"], color = 'steelblue', linestyle = 'dashed', label = "high_pass")
plt.scatter(years, changes_pass["low"], color = "#FFC000")
#plt.scatter(years, changes_pass["high"], color = 'steelblue')
plt.plot(years, changes_freight["low"], label = "low_freight", color = "#996633", linestyle = 'dotted')
#plt.plot(years, changes_freight["high"], label = "high_freight", color = 'midnightblue', linestyle = 'dashed')
plt.scatter(years, changes_freight["low"], color = "#996633")#, marker = '^')
#plt.scatter(years, changes_freight["high"], color = 'midnightblue')
sns.despine(trim=True)
#plt.legend()
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{euc}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'graph_all_low.svg', bbox_inches = 'tight')
plt.show()

#%% GIGA GRAPH HIGH
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
plt.plot(years, changes_LTdhn["high"], label = "high_dhn", color = "firebrick", linestyle = 'dashed')
plt.scatter(years, changes_LTdhn["high"], color = "firebrick")
plt.plot(years, changes_LTdec["high"], label = "high_dec", color = "#FF9999", linestyle = 'dashed')
plt.scatter(years, changes_LTdec["high"], color = "#FF9999")
plt.plot(years, changes_HT["high"], label = "high_ht", color = "#FF0000", linestyle = 'dashed')
plt.scatter(years, changes_HT["high"], color = "#FF0000")
plt.plot(years, changes_elec["high"], label = "high_elec", color = "#00B0F0", linestyle = 'dashed')
plt.scatter(years, changes_elec["high"], color = "#00B0F0")
plt.plot(years, changes_pass["high"], color = "#FFC000", linestyle = 'dashed', label = "high_pass")
plt.scatter(years, changes_pass["high"], color = "#FFC000")
plt.plot(years, changes_freight["high"], label = "high_freight", color = "#996633", linestyle = 'dashed')
plt.scatter(years, changes_freight["high"], color = "#996633")
sns.despine(trim=True)
#plt.legend()
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{euc}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
fig.savefig(folder_low + 'graph_all_high.svg', bbox_inches = 'tight')
plt.show()




