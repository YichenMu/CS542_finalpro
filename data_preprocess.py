#-*- coding=utf-8 -*-
#@File:data_preprocess.py
#@Software:PyCharm

import pandas as pd
import numpy as np
import csv

# with team=>identify the plays and coaches for every team in 2003
# without team=>to calculate the total efficiency of every player and coach

pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# To process and create new txt files to strip the space after the first column in the orginal files.
'''lines = open('databasebasketball2.0/player_playoffs_career.txt').readlines()
fp = open('databasebasketball2.0/Processed_player_playoffs_career.txt','w')
for s in lines:
    fp.write(s.replace(' ',''))
fp.close()
lines = open('databasebasketball2.0/player_playoffs.txt').readlines()
fp = open('databasebasketball2.0/Processed_player_playoffs.txt','w') 
for s in lines:
    fp.write(s.replace(' ',''))
fp.close()
lines = open('databasebasketball2.0/player_allstar.txt').readlines()
fp = open('databasebasketball2.0/Processed_player_allstar.txt','w') 
for s in lines:
    fp.write(s.replace(' ',''))
fp.close()'''
'''lines = open('databasebasketball2.0/coaches_career.txt').readlines()
fp = open('databasebasketball2.0/Processed_coaches_career.txt','w')
for s in lines:
    fp.write(s.replace(' ',''))
fp.close()'''
def process_fg(x):
    if x['fga']==0:
        return 0
    else:
        return x['fgm']/x['fga']

def process_ft(x):
    if x['fta']==0:
        return 0.0
    else:
        return x['ftm']/x['fta']
def process_pts(x):
    if x['minutes']==0:
        return 0.0
    else:
        return x['pts']/x['minutes']
def process_o_fg(x):
    if x['o_fga']==0:
        return 0.0
    else:
        return x['o_fgm']/x['o_fga']
def process_o_ft(x):
    if x['o_fta']==0:
        return 0.0
    else:
        return x['o_ftm']/x['o_fta']
def process_o_3p(x):
    if x['o_3pa']==0:
        return 0.0
    else:
        return x['o_3pm']/x['o_3pa']
def process_d_fg(x):
    if x['d_fga']==0:
        return 0.0
    else:
        return x['d_fgm']/x['d_fga']
def process_d_ft(x):
    if x['d_fta']==0:
        return 0.0
    else:
        return x['d_ftm']/x['d_fta']
def process_d_3p(x):
    if x['d_3pa']==0:
        return 0.0
    else:
        return x['d_3pm']/x['d_3pa']

player_season_stat_without_team=pd.read_csv('databasebasketball2.0/player_regular_season_career.txt', sep=',')
player_season_stat_with_team=pd.read_csv('databasebasketball2.0/player_regular_season.txt',sep=',')
player_season_stat_without_team['Season_Effic']=player_season_stat_without_team.apply(lambda x: (x['pts']+x['asts']+x['dreb']+x['oreb']+x['stl']+x['blk']-(x['fga']-x['fgm'])-(x['fta']-x['ftm'])-x['turnover'])/x['gp'],axis=1)
player_season_stat_without_team['Season_FG_rate']=player_season_stat_without_team.apply(lambda x: process_fg(x),axis=1)
player_season_stat_without_team['Season_FT_rate']=player_season_stat_without_team.apply(lambda x: process_ft(x),axis=1)
player_season_stat_without_team['Season_PTS_rate']=player_season_stat_without_team.apply(lambda x: process_pts(x),axis=1)
# print(player_season_stat_with_team.shape)
# print(player_season_stat_without_team)
temp1=[i for i in range(4,21)]
player_season_stat_without_team.drop(player_season_stat_without_team.columns[temp1],axis=1,inplace=True)
# print(temp1)
# player_season_stat_without_team.
# print(player_season_stat_without_team)


player_playoffs_stat_with_team=pd.read_csv('databasebasketball2.0/Processed_player_playoffs.txt',sep=',')
player_playoffs_stat_without_team=pd.read_csv('databasebasketball2.0/Processed_player_playoffs_career.txt',sep=',')
player_playoffs_stat_without_team['Playoffs_Effic']=player_playoffs_stat_without_team.apply(lambda x: (x['pts']+x['asts']+x['dreb']+x['oreb']+x['stl']+x['blk']-(x['fga']-x['fgm'])-(x['fta']-x['ftm'])-x['turnover'])/x['gp'],axis=1)
player_playoffs_stat_without_team['Playoffs_FG_rate']=player_playoffs_stat_without_team.apply(lambda x: process_fg(x),axis=1)
player_playoffs_stat_without_team['Playoffs_FT_rate']=player_playoffs_stat_without_team.apply(lambda x: process_ft(x),axis=1)
player_playoffs_stat_without_team['Playoffs_PTS_rate']=player_playoffs_stat_without_team.apply(lambda x: process_pts(x),axis=1)
# print(player_playoffs_stat_without_team.shape,player_playoffs_stat_with_team.shape)
player_playoffs_stat_without_team.drop(player_playoffs_stat_without_team.columns[temp1],axis=1,inplace=True)
# print(player_playoffs_stat_without_team)
# print(player_playoffs_stat_without_team)


player_allstar_stat_without_team=pd.read_csv('databasebasketball2.0/Processed_player_allstar.txt',sep=',')
# print(player_allstar_stat_without_team)
player_allstar_stat_without_team.fillna(0.0,inplace=True)
player_allstar_stat_without_team['Allstar_Effic']=player_allstar_stat_without_team.apply(lambda x: (x['pts']+x['asts']+x['dreb']+x['oreb']+x['stl']+x['blk']-(x['fga']-x['fgm'])-(x['fta']-x['ftm'])-x['turnover'])/x['gp'],axis=1)
temp2=[1,4]
temp2.extend([i for i in range(6,23)])
# print(temp2)
player_allstar_stat_without_team.drop(player_allstar_stat_without_team.columns[temp2],axis=1,inplace=True)
# print(player_allstar_stat_without_team)

player_stat_combine=pd.merge(player_season_stat_without_team,player_playoffs_stat_without_team,on=['ilkid','firstname','lastname','leag'])
# player_temp_combine=player_temp_combine.dropna()
# print(player_stat_combine)

# To output the dataframe outcome as the txt file.
'''player_stat_combine.to_csv('databasebasketball2.0/Processed_player_stat_combination.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')
'''
# Total stats of every player(Dataframe)
# player_total_stats=player_temp_combine[['ilkid','firstname','lastname','leag','Season_Effic','Playoffs_Effic']]
# player_total_stats.rename(columns={'Season_Effic_x':'Season_Effic','Season_Effic_y':'Playoffs_Effic'},inplace=True)
# print(player_total_stats)
player_season_stat_2003_with_team=player_season_stat_with_team[player_season_stat_with_team['year']==2003]
# print(player_season_stat_2003_with_team)
player_playoffs_stat_2003_with_team=player_playoffs_stat_with_team[player_playoffs_stat_with_team['year']==2003]
coaches_stat_with_team=pd.read_csv('databasebasketball2.0/coaches_season.txt',sep=',')
# print(coaches_stat_with_team)
coaches_stat_without_team=pd.read_csv('databasebasketball2.0/Processed_coaches_career.txt',sep=',')
# print(coaches_stat_without_team)
coaches_stat_without_team['Coach_Effic']=coaches_stat_without_team.apply(lambda x: (x['season_win']*2+x['season_loss']*0+x['playoff_win']*2+x['playoff_loss']*0)/(x['season_win']+x['season_loss']+x['playoff_win']+x['playoff_loss']),axis=1)
temp3=[i for i in range(1,7)]
# print(temp3)
coaches_stat_without_team.drop(coaches_stat_without_team.columns[temp3],axis=1,inplace=True)
# print(coaches_stat_without_team)
# To output the dataframe outcome as txt file
coaches_stat_without_team.columns=['Coachid','Coach_Effic']
# print(coaches_stat_without_team)
# coaches_stat_without_team.to_csv('databasebasketball2.0/Processed_coach_stat_with_effi.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')

# Stat about team
team_stat=pd.read_csv('databasebasketball2.0/team_season.txt',sep=',')
# a=list(team_stat['team'].unique())
# print(len(a))
# print(team_stat)
team_stat=team_stat.groupby('team',as_index=False).sum()
# print(team_stat)
team_stat['o_Effic']=team_stat.apply(lambda x: (x['o_pts']+x['o_asts']+x['o_dreb']+x['o_oreb']+x['o_stl']+x['o_blk']-(x['o_fga']-x['o_fgm'])-(x['o_fta']-x['o_ftm'])),axis=1)
team_stat['o_FG_rate']=team_stat.apply(lambda x: process_o_fg(x),axis=1)
team_stat['o_FT_rate']=team_stat.apply(lambda x: process_o_ft(x),axis=1)
team_stat['o_3P_rate']=team_stat.apply(lambda x: process_o_3p(x),axis=1)
team_stat['d_Effic']=team_stat.apply(lambda x: (x['d_pts']+x['d_asts']+x['d_dreb']+x['d_oreb']+x['d_stl']+x['d_blk']-(x['d_fga']-x['d_fgm'])-(x['d_fta']-x['d_ftm'])),axis=1)
team_stat['d_FG_rate']=team_stat.apply(lambda x: process_d_fg(x),axis=1)
team_stat['d_FT_rate']=team_stat.apply(lambda x: process_d_ft(x),axis=1)
team_stat['d_3P_rate']=team_stat.apply(lambda x: process_d_3p(x),axis=1)
team_stat['WIN_LOSE_rate']=team_stat.apply(lambda x: x['won']/(x['won']+x['lost']),axis=1)
# print(team_stat)
temp4=[i for i in range(1,35)]
team_stat.drop(team_stat.columns[temp4],axis=1,inplace=True)
team_stat.rename(columns={'team':'Team'},inplace=True)
# print(team_stat)
# To output the dataframe outcome as txt file
team_stat.to_csv('databasebasketball2.0/Processed_team_stat.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')

# fgm/fga
# ftm/fta
# 3pm/3pa
# pts/minutes

# Process the 2003_full_list_final.txt and create the processed file.
'''teamlist_2003=pd.read_csv('databasebasketball2.0/2003_full_list_final.txt',sep=',')
# teamlist_2003.dropna(inplace=True)
teamlist_2003=teamlist_2003.drop(columns=['Unnamed: 7'])
teamlist_2003.rename(columns={'Firstname':'CFirstname','Lastname':'CLastname','Firstname.1':'PFirstname','Lastname.1':'PLastname'},inplace=True)'''
'''teamlist_2003.to_csv('databasebasketball2.0/teamlist_stat_2003.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')'''
# print(teamlist_2003)

teamlist_stat_2003=pd.read_csv('databasebasketball2.0/teamlist_stat_2003.txt',sep=',')
# print(teamlist_stat_2003)
a=list(teamlist_stat_2003['Team'].unique())
# print(len(a))
# teamlist_stat_2003=teamlist_stat_2003.groupby('Team',as_index=False).sum()
# print(teamlist_stat_2003.shape)
Processed_coach_stat=pd.read_csv('databasebasketball2.0/Processed_coach_stat_with_effi.txt',sep=',')
teamlist_stat_2003=pd.merge(teamlist_stat_2003,Processed_coach_stat,on=['Coachid'])
Processed_team_stat=pd.read_csv('databasebasketball2.0/Processed_team_stat.txt',sep=',')
teamlist_stat_2003=pd.merge(teamlist_stat_2003,Processed_team_stat,on=['Team'])
# print(teamlist_stat_2003)



# To merge, filter, calculate some new features to create the main training dataset we will use in data_learn.py file later.

team_data = pd.read_csv('databasebasketball2.0/team_season.txt', keep_default_na=False)
test_2003_data=team_data[team_data['year']==2003]
player_data = pd.read_csv('databasebasketball2.0/player_regular_season.txt', keep_default_na=False)
coach_data = pd.read_csv('databasebasketball2.0/coaches_season.txt', keep_default_na=False)
final_team_data=team_data[~team_data.year.isin([2003,2004])]
print(final_team_data)
# print(coach_data)
print("\n")
# print(team_data)
print("\n")
tmp_data = pd.merge(final_team_data,coach_data,how='inner',on=['team','year'])
# print(tmp_data)
print("\n")
final_data = pd.merge(tmp_data,player_data,how='inner',on=['team','year'])
# print(final_data)
temp5=[2,38,39,42,43,45,46]
# To delete some non-numeric columns
final_data.drop(final_data.columns[temp5],axis=1,inplace=True)
# print(final_data)
final_data.rename(columns={'leag_y':'leag'},inplace=True)
final_data['turnover']=final_data['turnover'].astype(float)
# Calculate and create some new feature columns
final_data['T_o_Effic']=final_data.apply(lambda x: (x['o_pts']+x['o_asts']+x['o_dreb']+x['o_oreb']+x['o_stl']+x['o_blk']-(x['o_fga']-x['o_fgm'])-(x['o_fta']-x['o_ftm'])),axis=1)
final_data['T_o_FG_rate']=final_data.apply(lambda x: process_o_fg(x),axis=1)
final_data['T_o_FT_rate']=final_data.apply(lambda x: process_o_ft(x),axis=1)
final_data['T_o_3P_rate']=final_data.apply(lambda x: process_o_3p(x),axis=1)
final_data['T_d_Effic']=final_data.apply(lambda x: (x['d_pts']+x['d_asts']+x['d_dreb']+x['d_oreb']+x['d_stl']+x['d_blk']-(x['d_fga']-x['d_fgm'])-(x['d_fta']-x['d_ftm'])),axis=1)
final_data['T_d_FG_rate']=final_data.apply(lambda x: process_d_fg(x),axis=1)
final_data['T_d_FT_rate']=final_data.apply(lambda x: process_d_ft(x),axis=1)
final_data['T_d_3P_rate']=final_data.apply(lambda x: process_d_3p(x),axis=1)
final_data['T_WIN_LOSE_rate']=final_data.apply(lambda x: x['won']/(x['won']+x['lost']),axis=1)
final_data['Coach_Effic']=final_data.apply(lambda x: (x['season_win']*2+x['season_loss']*0)/(x['season_win']+x['season_loss']),axis=1)
final_data['P_Season_Effic']=final_data.apply(lambda x: (x['pts']+x['asts']+x['dreb']+x['oreb']+x['stl']+x['blk']-(x['fga']-x['fgm'])-(x['fta']-x['ftm'])-x['turnover'])/x['gp'],axis=1)
final_data['P_Season_FG_rate']=final_data.apply(lambda x: process_fg(x),axis=1)
final_data['P_Season_FT_rate']=final_data.apply(lambda x: process_ft(x),axis=1)
final_data['P_Season_PTS_rate']=final_data.apply(lambda x: process_pts(x),axis=1)
# Save the file for later use
final_data.to_csv('databasebasketball2.0/Processed_final_data.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')

# print(final_data)

# To merge, filter, calculate some new features to create the testing dataset
tmp1_data = pd.merge(test_2003_data,coach_data,how='inner',on=['team','year'])
test_2003_data = pd.merge(tmp1_data,player_data,how='inner',on=['team','year'])
# print(test_2003_data)
test_2003_data.drop(test_2003_data.columns[temp5],axis=1,inplace=True)
# print(final_data)
test_2003_data.rename(columns={'leag_y':'leag'},inplace=True)
test_2003_data['turnover']=test_2003_data['turnover'].astype(float)
test_2003_data['T_o_Effic']=test_2003_data.apply(lambda x: (x['o_pts']+x['o_asts']+x['o_dreb']+x['o_oreb']+x['o_stl']+x['o_blk']-(x['o_fga']-x['o_fgm'])-(x['o_fta']-x['o_ftm'])),axis=1)
test_2003_data['T_o_FG_rate']=test_2003_data.apply(lambda x: process_o_fg(x),axis=1)
test_2003_data['T_o_FT_rate']=test_2003_data.apply(lambda x: process_o_ft(x),axis=1)
test_2003_data['T_o_3P_rate']=test_2003_data.apply(lambda x: process_o_3p(x),axis=1)
test_2003_data['T_d_Effic']=test_2003_data.apply(lambda x: (x['d_pts']+x['d_asts']+x['d_dreb']+x['d_oreb']+x['d_stl']+x['d_blk']-(x['d_fga']-x['d_fgm'])-(x['d_fta']-x['d_ftm'])),axis=1)
test_2003_data['T_d_FG_rate']=test_2003_data.apply(lambda x: process_d_fg(x),axis=1)
test_2003_data['T_d_FT_rate']=test_2003_data.apply(lambda x: process_d_ft(x),axis=1)
test_2003_data['T_d_3P_rate']=test_2003_data.apply(lambda x: process_d_3p(x),axis=1)
test_2003_data['T_WIN_LOSE_rate']=test_2003_data.apply(lambda x: x['won']/(x['won']+x['lost']),axis=1)
test_2003_data['Coach_Effic']=test_2003_data.apply(lambda x: (x['season_win']*2+x['season_loss']*0)/(x['season_win']+x['season_loss']),axis=1)
test_2003_data['P_Season_Effic']=test_2003_data.apply(lambda x: (x['pts']+x['asts']+x['dreb']+x['oreb']+x['stl']+x['blk']-(x['fga']-x['fgm'])-(x['fta']-x['ftm'])-x['turnover'])/x['gp'],axis=1)
test_2003_data['P_Season_FG_rate']=test_2003_data.apply(lambda x: process_fg(x),axis=1)
test_2003_data['P_Season_FT_rate']=test_2003_data.apply(lambda x: process_ft(x),axis=1)
test_2003_data['P_Season_PTS_rate']=test_2003_data.apply(lambda x: process_pts(x),axis=1)
print(test_2003_data)
# Save the file for later use
test_2003_data.to_csv('databasebasketball2.0/Processed_test_2003_data.txt',sep=',',index=False,quoting=csv.QUOTE_NONE,escapechar=',')



