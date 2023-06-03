# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:43:04 2023

@author: julia

Graphs for the local sensitivity analyses: variations concerning the resources
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
folder_low = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_3_freight_changes_low/"
subfolder_low = "output_3_freight_changes_low/"
folder_nom = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_base_case/"
subfolder_nom = "output_base_case/"
folder_high = "C:/Users/julia/OneDrive - UCL/UCLouvain/M2/Mémoire/ENERGYSCOPE/LSA/output_8_freight_changes_high/"
subfolder_high = "output_8_freight_changes_high/"

folders = [folder_low, folder_nom, folder_high]
subfolders = [subfolder_low, subfolder_nom, subfolder_high]

#%%
cases = ["low", "nom", "high"]
dicofdf_res = {}
for c in range(len(cases)) : 
    df = pd.read_excel(folders[c]+subfolders[c]+"resources.xlsx", sheet_name=2).set_index("Name")
    total = np.zeros(7)
    for j in range(len(years)) : 
        for i in range(len(df)) : 
            total[j] += df.iloc[i,j]
    df.loc[len(df)] = total
    df.rename(index={len(df)-1 : "Total"}, inplace = True)
    dicofdf_res[cases[c]] = df
#%% total change in the mix compared to nominal case
change_tot_low = np.zeros(7)
change_tot_high = np.zeros(7)

for y in range(len(years)) :
    change_tot_low[y] = (dicofdf_res["nom"].iloc[-1,y] - dicofdf_res["low"].iloc[-1,y])/dicofdf_res["nom"].iloc[-1,y]
    change_tot_high[y] = (dicofdf_res["nom"].iloc[-1,y] - dicofdf_res["high"].iloc[-1,y])/dicofdf_res["nom"].iloc[-1,y]

#%% Compute percentage of each resource in the mix
dicofdf_perc_res = {}
for c in range(len(cases)) : 
    df = dicofdf_res[cases[c]]
    df_perc = pd.DataFrame(columns = years)

    for i in range(len(df)-1) : 
        perc = np.zeros(7)
        for y in range(len(years)) : 
            perc[y] = df.iloc[i,y]/df.iloc[-1,y]
        df_perc.loc[len(df_perc)] = perc
        df_perc.rename(index={len(df_perc)-1 : df.index[i]}, inplace = True)
    
    dicofdf_perc_res[cases[c]] = df_perc

#%% compute the difference in the percentage of the mix compared to nominal case
dicofdf_diff = {}

df_diff_low = pd.DataFrame(columns = years)
df_diff_high = pd.DataFrame(columns = years)

for i in range(len(dicofdf_perc_res["nom"])) : 
    diff_low_i = []
    diff_high_i = []
    for y in range(len(years)) : 
        diff_low = -dicofdf_perc_res["nom"].iloc[i,y] + dicofdf_perc_res["low"].iloc[i,y]
        diff_high = -dicofdf_perc_res["nom"].iloc[i,y] + dicofdf_perc_res["high"].iloc[i,y]
        diff_low_i.append(diff_low)
        diff_high_i.append(diff_high)
    df_diff_low.loc[len(df_diff_low)] = diff_low_i
    df_diff_low.rename(index = {len(df_diff_low)-1 : dicofdf_perc_res["nom"].index[i]}, inplace = True)        
    df_diff_high.loc[len(df_diff_high)] = diff_high_i
    df_diff_high.rename(index = {len(df_diff_high)-1 : dicofdf_perc_res["nom"].index[i]}, inplace = True)        

#%% draw changes in the resources, higher than 1%

res_to_plot = []
for i in range(len(df_diff_high)) : 
    res = df_diff_high.index[i]
    toplot = False 
    for y in range(len(years)) : 
        if np.abs(df_diff_high.iloc[i,y]) >= 0.01 or np.abs(df_diff_low.iloc[i,y]) >= 0.01 : 
            toplot = True 
    if toplot == True : 
        res_to_plot.append(i)


colors = {'URANIUM' : '#FF9999', 'WASTE' : '#808000', 'GAS' : '#FFC000', 'RES_SOLAR' : '#FFFF00', 
          'WET_BIOMASS' : '#336600', 'WOOD' : '#996633', 'GAS_RE' : '#FFE697', 'COAL' : '#000000',
          'H2' : '#BD5BA6', 'AMMONIA_RE' : '#3F47D0', 'DIESEL' : '#A5A5A5', 'GASOLINE' : '#7F7F7F',
          'LFO' : '#7030A0', 'RES_WIND' : "#00B050", 'ELECTRICITY' : "#00B0F0", 'BIOFUELS' : "rosybrown"}
#%% LOW
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in res_to_plot : 
    res = df_diff_high.index[i]
    plt.plot(years, df_diff_low.loc[res], color = colors[res], linestyle = 'dotted', label = res)
    plt.scatter(years, df_diff_low.loc[res], color = colors[res])
#plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{res}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
#fig.savefig(folder_low + 'graph_resources_change_low_notabs.svg', bbox_inches = 'tight')
plt.show()

#%% HIGH
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()
for i in res_to_plot : 
    res = df_diff_high.index[i]
    plt.plot(years, df_diff_high.loc[res], color = colors[res], linestyle = 'dashed', label = res)
    plt.scatter(years, df_diff_high.loc[res], color = colors[res])
#plt.legend()
sns.despine(trim = True)
ax.set_xlabel('Year', rotation=0, ha='right', va='bottom',fontname='Arial',size=30)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("$\Delta share_{res}$", rotation=0, ha='left', va='bottom',fontname='Arial',size=30)
ax.yaxis.set_label_coords(-0.1, 1.)
#fig.savefig(folder_low + 'graph_resources_change_high_notabs.svg', bbox_inches = 'tight')
plt.show()