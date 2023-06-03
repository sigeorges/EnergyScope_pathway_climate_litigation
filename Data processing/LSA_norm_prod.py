# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:19:38 2023

@author: julia

Graphs for the local sensitivity analyses: normalized production indicator

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

#toreplace
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
    
#%%

colors = {" NUCLEAR ":"deeppink", " CCGT ":"darkorange", " CCGT_AMMONIA ":"slateblue", " COAL_US " : "black", " COAL_IGCC " : "dimgray", " PV " : "yellow", " WIND_ONSHORE " : "lawngreen", " WIND_OFFSHORE " : "green", " HYDRO_RIVER " : "blue", " GEOTHERMAL " : "firebrick", " ELECTRICITY " : "dodgerblue"}
colors.update({" DHN_HP_ELEC " : "blue", " DHN_COGEN_GAS " : "orange", " DHN_COGEN_WOOD " : "sandybrown", " DHN_COGEN_WASTE " : "olive", " DHN_COGEN_WET_BIOMASS " : "seagreen", " DHN_COGEN_BIO_HYDROLYSIS " : "springgreen", " DHN_BOILER_GAS " : "darkorange", " DHN_BOILER_WOOD " : "sienna", " DHN_BOILER_OIL " : "blueviolet", " DHN_DEEP_GEO " : "firebrick", " DHN_SOLAR " : "gold", " DEC_HP_ELEC " : "cornflowerblue", " DEC_THHP_GAS " : "lightsalmon", " DEC_COGEN_GAS " : "goldenrod", " DEC_COGEN_OIL " : "mediumpurple", " DEC_ADVCOGEN_GAS " : " burlywood ", " DEC_ADVCOGEN_H2 " : "violet", " DEC_BOILER_GAS " : "moccasin", " DEC_BOILER_WOOD " : "peru", " DEC_BOILER_OIL " : "darkorchid", " DEC_SOLAR " : "yellow", " DEC_DIRECT_ELEC " : "deepskyblue"})
colors.update({" IND_COGEN_GAS ":"orange", " IND_COGEN_WOOD ":"peru", " IND_COGEN_WASTE " : "olive", " IND_BOILER_GAS " : "moccasin", " IND_BOILER_WOOD " : "goldenrod", " IND_BOILER_OIL " : "blueviolet", " IND_BOILER_COAL " : "black", " IND_BOILER_WASTE " : "olivedrab", " IND_DIRECT_ELEC " : "royalblue"})

#%% all sectors except mobility

giga_df_nom = pd.concat([
dicofdf_elec["nom"].drop(index = 'Total'),
dicofdf_HT["nom"].drop(index = 'Total'),
dicofdf_LTdec["nom"].drop(index = 'Total'),
dicofdf_LTdhn["nom"].drop(index = 'Total')])

c = giga_df_nom.columns
dic_col = {}
for i in range(len(c)) : 
    dic_col[c[i]] = years[i]

giga_df_nom = giga_df_nom.groupby(giga_df_nom.index).sum()

max_tech = giga_df_nom.values.max()

max_row = giga_df_nom[giga_df_nom.values == max_tech]

giga_df_nom = giga_df_nom/max_tech

giga_df_nom = giga_df_nom[giga_df_nom.max(axis=1) > 0.5]
giga_df_nom.rename(columns = dic_col, inplace = True)

color = []
for i in range(len(giga_df_nom)) : 
    color.append(colors[giga_df_nom.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_nom.T.plot(legend=False, lw=3, color=color, ax = ax)
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_nom.svg')

#%% (min)
giga_df_low = pd.concat([
dicofdf_elec["low"].drop(index = 'Total'),
dicofdf_HT["low"].drop(index = 'Total'),
dicofdf_LTdec["low"].drop(index = 'Total'),
dicofdf_LTdhn["low"].drop(index = 'Total')])

giga_df_low = giga_df_low.groupby(giga_df_low.index).sum()

max_tech = giga_df_low.values.max()

max_row = giga_df_low[giga_df_low.values == max_tech]

giga_df_low = giga_df_low/max_tech

giga_df_low = giga_df_low[giga_df_low.max(axis=1) > 0.5]
giga_df_low.rename(columns = dic_col, inplace = True)

color = []
for i in range(len(giga_df_low)) : 
    color.append(colors[giga_df_low.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_low.T.plot(legend=True, lw=3, color=color, ax = ax, linestyle = 'dotted')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_low_2_7.svg')


#%% (max)

giga_df_high = pd.concat([
dicofdf_elec["high"].drop(index = 'Total'),
dicofdf_HT["high"].drop(index = 'Total'),
dicofdf_LTdec["high"].drop(index = 'Total'),
dicofdf_LTdhn["high"].drop(index = 'Total')])

giga_df_high = giga_df_high.groupby(giga_df_high.index).sum()

max_tech = giga_df_high.values.max()

max_row = giga_df_high[giga_df_high.values == max_tech]

giga_df_high = giga_df_high/max_tech

giga_df_high = giga_df_high[giga_df_high.max(axis=1) > 0.5]
giga_df_high.rename(columns = dic_col, inplace = True)
color = []
for i in range(len(giga_df_high)) : 
    color.append(colors[giga_df_high.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_high.T.plot(legend=True, lw=3, color=color, ax = ax, linestyle = 'dashed')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_high_2_7.svg')

#%%

giga_df_high.T.plot(legend=True, color = colors, lw=3)
giga_df_low.T.plot(legend=True, color = colors, lw=3, linestyle = 'dotted')
giga_df_nom.T.plot(legend=True, color = colors, lw=3, linestyle = 'dashed')
"""
for i in range(len(giga_df_nom)) :
    plt.plot(years, giga_df_nom.iloc[i], color = colors[giga_df_nom.index[i]], label= giga_df_nom.index[i])
for i in range(len(giga_df_low)) :
    plt.plot(years, giga_df_low.iloc[i], color = colors[giga_df_low.index[i]], label= giga_df_low.index[i], linestyle = 'dotted')
for i in range(len(giga_df_high)) :
    plt.plot(years, giga_df_high.iloc[i], color = colors[giga_df_high.index[i]], label= giga_df_high.index[i], linestyle = 'dashed')
#plt.legend()
plt.show()"""

#%% MOB PASS (min)
colors_mob = {" TRAMWAY_TROLLEY " : "dodgerblue", " BUS_COACH_DIESEL " : "dimgrey", " BUS_COACH_HYDIESEL " : "gray", " BUS_COACH_CNG_STOICH " : "orange", " BUS_COACH_FC_HYBRIDH2 " : "violet", " TRAIN_PUB " : "blue", " CAR_GASOLINE " : "black", " CAR_DIESEL " : "lightgray", " CAR_NG " : "moccasin", " CAR_METHANOL ":"orchid", " CAR_HEV " : "salmon", " CAR_PHEV " : "lightsalmon", " CAR_BEV " : "deepskyblue", " CAR_FUEL_CELL " : "magenta"}

giga_df_pass_low = dicofdf_pass["low"].drop(index = 'Total')

max_tech = giga_df_pass_low.values.max()

max_row = giga_df_pass_low[giga_df_pass_low.values == max_tech]

giga_df_pass_low = giga_df_pass_low/max_tech

giga_df_pass_low = giga_df_pass_low[giga_df_pass_low.max(axis=1) > 0.05]

c = giga_df_pass_low.columns

giga_df_pass_low.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)


color = []
for i in range(len(giga_df_pass_low)) : 
    color.append(colors_mob[giga_df_pass_low.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_pass_low.T.plot(legend=False, lw=3, color=color, ax = ax, linestyle = 'dotted')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_pass_low_4_5.svg')

#%% (max)
giga_df_pass_high = dicofdf_pass["high"].drop(index = 'Total')

max_tech = giga_df_pass_high.values.max()

max_row = giga_df_pass_high[giga_df_pass_high.values == max_tech]

giga_df_pass_high = giga_df_pass_high/max_tech

giga_df_pass_high = giga_df_pass_high[giga_df_pass_high.max(axis=1) > 0.05]

c = giga_df_pass_high.columns

giga_df_pass_high.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)

color = []
for i in range(len(giga_df_pass_high)) : 
    color.append(colors_mob[giga_df_pass_high.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_pass_high.T.plot(legend=False, lw=3, color=color, ax = ax, linestyle = 'dashed')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_pass_high_4_5.svg')

#%% nom

giga_df_pass_nom = dicofdf_pass["nom"].drop(index = 'Total')

max_tech = giga_df_pass_nom.values.max()

max_row = giga_df_pass_nom[giga_df_pass_nom.values == max_tech]

giga_df_pass_nom = giga_df_pass_nom/max_tech

giga_df_pass_nom = giga_df_pass_nom[giga_df_pass_nom.max(axis=1) > 0.05]

c = giga_df_pass_nom.columns

giga_df_pass_nom.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)


color = []
for i in range(len(giga_df_pass_nom)) : 
    color.append(colors_mob[giga_df_pass_nom.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_pass_nom.T.plot(legend=False, lw=3, color=color, ax = ax)
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_pass_nom_2.svg')

#%% MOB FREIGHT (min)

colors_freight = {" TRAIN_FREIGHT " : "royalblue", " BOAT_FREIGHT_DIESEL " : "dimgrey", " BOAT_FREIGHT_NG " : "darkorange", " BOAT_FREIGHT_METHANOL " : "fuchsia", " TRUCK_DIESEL " : "darkgrey", " TRUCK_FUEL_CELL " : "violet", " TRUCK_ELEC " : "dodgerblue", " TRUCK_NG " : "moccasin", " TRUCK_METHANOL " : "orchid"}
giga_df_freight_low = dicofdf_freight["low"].drop(index = 'Total')

max_tech = giga_df_freight_low.values.max()

max_row = giga_df_freight_low[giga_df_pass_low.values == max_tech]

giga_df_freight_low = giga_df_freight_low/max_tech

giga_df_freight_low = giga_df_freight_low[giga_df_freight_low.max(axis=1) > 0.05]

c = giga_df_freight_low.columns

giga_df_freight_low.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)

color = []
for i in range(len(giga_df_freight_low)) : 
    color.append(colors_freight[giga_df_freight_low.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_freight_low.T.plot(legend=False, lw=3, color=color, ax = ax, linestyle = 'dotted')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_freight_low_3_8.svg')
plt.show()

#%% (max)
giga_df_freight_high = dicofdf_freight["high"].drop(index = 'Total')

max_tech = giga_df_freight_high.values.max()

max_row = giga_df_freight_high[giga_df_pass_high.values == max_tech]

giga_df_freight_high = giga_df_freight_high/max_tech

giga_df_freight_high = giga_df_freight_high[giga_df_freight_high.max(axis=1) > 0.05]

c = giga_df_freight_high.columns

giga_df_freight_high.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)

color = []
for i in range(len(giga_df_freight_high)) : 
    color.append(colors_freight[giga_df_freight_high.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_freight_high.T.plot(legend=False, lw=3, color=color, ax = ax, linestyle = 'dashed')
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_freight_high_3_8.svg')
plt.show()

#%% (nom)
giga_df_freight_nom = dicofdf_freight["nom"].drop(index = 'Total')

max_tech = giga_df_freight_nom.values.max()

max_row = giga_df_freight_nom[giga_df_freight_nom.values == max_tech]

giga_df_freight_nom = giga_df_freight_nom/max_tech

giga_df_freight_nom = giga_df_freight_nom[giga_df_freight_nom.max(axis=1) > 0.05]

c = giga_df_freight_nom.columns

giga_df_freight_nom.rename(columns = {c[0] : years[0], c[1] : years[1], c[2] : years[2], c[3] : years[3], 
                                   c[4] : years[4], c[5] : years[5], c[6] : years[6]}, inplace = True)


color = []
for i in range(len(giga_df_freight_nom)) : 
    color.append(colors_freight[giga_df_freight_nom.index[i]])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot()
giga_df_freight_nom.T.plot(legend=False, lw=3, color=color, ax = ax)
sns.despine(trim = True)
#plt.savefig(folder_low+'normalized_importance_freight_nom_2.svg')