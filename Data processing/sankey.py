# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:10:01 2023

@author: Simon Georges (UCLouvain)
"""

# %% Importing packages
import pandas as pd
import numpy as np

# %% SankeyMATIC user manual

# // Enter Flows between Nodes, like this:
# //         Source [AMOUNT] Target

# Wages [1500] Budget
# Other [250] Budget

# Budget [450] Taxes
# Budget [420] Housing
# Budget [400] Food
# Budget [295] Transportation
# Budget [25] Savings

# // You can set a Node's color, like this:
# :Budget #708090
# //            ...or a color for a single Flow:
# Budget [160] Other Necessities #0F0

# %% Importing data
CO2_col = ['CO2_ATM', 'CO2_INDUSTRY', 'CO2_CAPTURED']
EUD_col = ['ELECTRICITY', 'HEAT_HIGH_T', 'HEAT_LOW_T_DHN', 'HEAT_LOW_T_DECEN', 'MOB_PUBLIC', 'MOB_PRIVATE', 'MOB_FREIGHT_RAIL', 'MOB_FREIGHT_ROAD', 'MOB_FREIGHT_BOAT', 'HVC']
RES_col = ['RES_WIND', 'RES_SOLAR', 'RES_HYDRO', 'RES_GEO']

CO2_row = [' CO2_ATM ', ' CO2_INDUSTRY ', ' CO2_CAPTURED ']
infra_row = [' EFFICIENCY ', ' DHN ', ' GRID ']
storage_row = [' PHS ', ' BATT_LI ', ' BEV_BATT ', ' PHEV_BATT ', ' TS_DEC_HP_ELEC ',
               ' TS_DEC_DIRECT_ELEC ', ' TS_DHN_DAILY ', ' TS_DHN_SEASONAL ', ' TS_DEC_THHP_GAS ',
               ' TS_DEC_COGEN_GAS ', ' TS_DEC_COGEN_OIL ', ' TS_DEC_ADVCOGEN_GAS ',
               ' TS_DEC_ADVCOGEN_H2 ', ' TS_DEC_BOILER_GAS ', ' TS_DEC_BOILER_WOOD ',
               ' TS_DEC_BOILER_OIL ', ' TS_HIGH_TEMP ', ' GAS_STORAGE ', ' H2_STORAGE ',
               ' CO2_STORAGE ', ' GASOLINE_STORAGE ', ' DIESEL_STORAGE ', ' METHANOL_STORAGE ',
               ' AMMONIA_STORAGE ', ' LFO_STORAGE ']
EUD_row = [' END_USES_DEMAND ']

year = '2020'

path = '..\STEP_2_Pathway_Model\output\YEAR_' + year
file = '\year_balance.txt'

year_bal = pd.read_table(path+file, index_col='Tech')

EUD = year_bal.loc[EUD_row].copy(deep=True)

year_bal.drop(columns=CO2_col+RES_col, inplace=True)
year_bal.drop(index=CO2_row+EUD_row, inplace=True)
year_bal.dropna(axis=1, inplace=True)

flow_out_dic = {'CAR': 'Mob private', 'FREIGHT': 'Freight', 'TRUCK': 'Freight', 'PUB': 'Mob public', 'TRAMWAY': 'Mob public', 'BUS': 'Mob public'}
flow_in_dic = {'BOILER': 'Boilers', 'COGEN': 'CHP', 'HP': 'HPs', 'THHP': 'HPs'}
import_res = ['ELECTRICITY', 'GAS', 'GASOLINE', 'DIESEL']

eu_name_format = {'ELECTRICITY': 'ELECTRICITY', 'HEAT_HIGH_T': 'Heat HT', 'HEAT_LOW_T_DHN': 'Heat LT DHN',
                  'HEAT_LOW_T_DECEN': 'Heat LT Dec', 'MOB_PUBLIC': 'Mob public', 'MOB_PRIVATE': 'Mob private',
                  'MOB_FREIGHT_RAIL': 'Freight', 'MOB_FREIGHT_BOAT': 'Freight', 'MOB_FREIGHT_ROAD': 'Freight'}

EUD_colors = {'Heat LT Dec': "#FF9999", 'Heat LT DHN': "#B22222", 'Heat HT': "#FF0000", 'Elec demand': "#00B0F0", 'Freight': "#996633", 'Mob public': "#FFC000", 'Mob private': "#BF9000"}

colors = ["#FFC592", "#006CB5", "#FFFF00", "#00B050", "#00B050", "#996633", "#336600", "#FFE697", "#FFC000", "#FFC000", "#808000", "#000000", "#FF9999", "#00B0F0", "#00B0F0", "#BD5BA6", "#3F47D0", "#A5A5A5", "#7F7F7F", "#7030A0", "#BC8F8F"]
res = ['HYDRO_RIVER', 'BIODIESEL', 'PV', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'WOOD', 'WET_BIOMASS', 'GAS_RE', 'GAS', 'GAS_import', 'WASTE', 'COAL', 'URANIUM', 'ELECTRICITY', 'ELECTRICITY_import', 'H2', 'AMMONIA_RE', 'DIESEL', 'GASOLINE', 'LFO', 'BIOETHANOL']
res_color_dic = {res[i]: colors[i] for i in range(len(res))}
# %% Printing in SankeyMATIC format
res_printed = []

for layer in year_bal.columns:
    if 'MOB' in layer.split('_'):
        pass
    else:
        lay = layer
        if layer in eu_name_format.keys():
            lay = eu_name_format[layer]
        for el in year_bal[layer].index:
            val = round(year_bal.loc[el, layer]/1e3, 2) #[TWh]
            val_to_print = '[' + str(abs(val)) + ']'
            
            elem = el.strip()
            
            if el in storage_row:
                elem = 'Storage losses'
            else:
                for key in flow_in_dic.keys():
                    if key in elem.split('_') and 'TS' not in elem.split('_'):
                        elem = flow_in_dic[key]
            if val > 1e-3:
                if elem != layer:
                    print(elem, val_to_print, lay)
                elif elem in import_res:
                    print(elem+'_import', val_to_print, lay)
                res_printed.append(elem)
            elif val < -1e-3:
                printed = False
                for key in flow_out_dic.keys():
                    if key in elem.split('_'):
                        print(lay, val_to_print, flow_out_dic[key])
                        printed = True
                if not printed:
                    print(lay, val_to_print, elem)
                res_printed.append(elem)
                
print('ELECTRICITY', '[' + str(round(EUD['ELECTRICITY'].values[0]/1e3, 2)) + ']', 'Elec demand')
var = round(EUD['ELECTRICITY'].values[0]/1e3, 2)
print('\n')
for key in EUD_colors.keys():
    print(':'+key, EUD_colors[key])
print('\n')
for key in res_printed:
    if key in res_color_dic.keys():
        print(':'+key, res_color_dic[key])