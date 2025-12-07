#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 20:27:22 2025

@author: jamesshoenhair
"""

#Import Libraries for Analysis
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
  
  
#Import Data Set
  
import ucimlrepo
  
from ucimlrepo import fetch_ucirepo, list_available_datasets
  
fetch_ucirepo(name='CDC Diabetes Health Indicators')
  
  
#Exploratory Data Analysis
  
dataset = fetch_ucirepo(name='CDC Diabetes Health Indicators')
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
  
print(f"DataFrame created: {df.shape[0]:,} rows × {df.shape[1]} columns")
  
  
print("First 20 rows")
df.head()
  
#Check for Missing Values\n",
  
print("Missing values per column")
print(df.isnull().sum())
  
rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Rows with missing values: {rows_with_missing:,}")
  
total_missing = df.isnull().sum().sum()
print(f"Total missing values: {total_missing:,}")
  
missing_summary = pd.DataFrame()
Missing_Count: df.isnull().sum()
Missing_Percentage: (df.isnull().sum() / len(df)) * 100
  
print("Missing Values Summary:")
print(missing_summary)
  
#Trimmed Columns
  
df_trimmed = df[['Age', 'Sex', 'Education', 'Income']].copy()
print(f"df_trimmed: {df_trimmed.shape}")
  
#Clean DF
  
df_clean = df_trimmed.dropna()
print(f"df_clean: {df_clean.shape}")
  
df_plot = df_clean
print(f"df_plot: {df_plot.shape}")
  
  
#Initial Visualizations
  
df_plot = df_clean
  
  
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_plot['Age'], bins=30, kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
  
sns.countplot(x='Sex', data=df_plot, ax=axes[0,1])
axes[0,1].set_title('Sex Distribution')
  
sns.countplot(x='Education', data=df_plot, ax=axes[1,0])
axes[1,0].set_title('Education Distribution')
  
sns.histplot(df_plot['Income'], bins=30, kde=True, ax=axes[1,1])
axes[1,1].set_title('Income Distribution')
  
plt.tight_layout()
plt.show()

#Encode Demographic Variables

#Encode Sex as a Binary Variable

df_processed = df.copy()

print("ENCODING SEX AS BINARY (0 = Female, 1 = Male)")

if 'Sex' in df_processed.columns:
    unique_vals = sorted(df_processed['Sex'].unique())
    print(f"\nOriginal Sex values: {unique_vals}")
    
    if not all(val in [0, 1] for val in unique_vals):
        df_processed['Sex'] = pd.Categorical(df_processed['Sex']).codes
        print(f"Sex encoded as binary: [0, 1]")
    else:
        print(f"Sex already encoded as binary: [0, 1]")
        

print(f"\nSex encoding:")
print(f"  0 = Female")
print(f"  1 = Male")
print(f"\nSex distribution:")
sex_dist = df_processed['Sex'].value_counts().sort_index()
for val, count in sex_dist.items():
   label = "Female" if val == 0 else "Male"
   pct = (count / len(df_processed)) * 100
   print(f"  {label} ({val}): {count:,} ({pct:.2f}%)")


#Keep Education as an Ordinal Variable

if 'Education' in df_processed.columns:
    print(f"\Education kept as ORDINAL variable (no encoding applied)")
    print(f"\nEducation details:")
    print(f"  Range: {df_processed['Education'].min()} to {df_processed['Education'].max()}")
    print(f"  Unique values: {sorted(df_processed['Education'].unique())}")
    print(f"  Data type: {df_processed['Education'].dtype}")
    
    print(f"\nEducation level interpretation (typical CDC coding):")
    print(f"  1 = Never attended or kindergarten only")
    print(f"  2 = Elementary (Grades 1-8)")
    print(f"  3 = Some high school (Grades 9-11)")
    print(f"  4 = High school graduate (Grade 12 or GED)")
    print(f"  5 = Some college or technical school")
    print(f"  6 = College graduate (4+ years)")
    
    print(f"\nEducation distribution:")
    edu_dist = df_processed['Education'].value_counts().sort_index()
    for val, count in edu_dist.items():
        pct = (count / len(df_processed)) * 100
        print(f"  Level {val}: {count:,} ({pct:.2f}%)")
else:
    print("\n⚠ Education variable not found in dataset")
    

#Missing Data in Encoded Variables

lifestyle_vars = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                 'Smoker', 'MentHlth', 'PhysHlth']

existing_lifestyle_vars = [var for var in lifestyle_vars if var in df_processed.columns]
print(f"\nLifestyle variables in dataset: {existing_lifestyle_vars}")

print("\nMissing values in lifestyle variables:")
has_missing = False
for var in existing_lifestyle_vars:
    missing = df_processed[var].isnull().sum()
    missing_pct = (missing / len(df_processed)) * 100
    if missing > 0:
        print(f"  {var}: {missing} ({missing_pct:.2f}%)")
        has_missing = True

if not has_missing:
    print("No missing values in lifestyle variables")
    

if df_processed[var].nunique() <= 2:
                mode_val = df_processed[var].mode()[0]
                df_processed[var].fillna(mode_val, inplace=True)
                print(f"  {var}: Filled with mode ({mode_val})")
                
                
#Key Demographic Variables

demographic_vars = []
if 'Sex' in df.columns: demographic_vars.append('Sex')
if 'Age' in df.columns: demographic_vars.append('Age')
if 'Education' in df.columns: demographic_vars.append('Education')
if 'Income' in df.columns: demographic_vars.append('Income')
                

#Univariate Analysis (Demographic Variables)

fig = plt.figure(figsize=(20, 16))

for idx, var in enumerate(demographic_vars, 1):
    if var in df_processed.columns:
        
#Histogram

        plt.subplot(4, 4, (idx-1)*4 + 1)
        sns.histplot(df_processed[var], bins=30, kde=True, color='steelblue')
        plt.title(f'{var} - Histogram with KDE', fontsize=12, fontweight='bold')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        
#Box Plot
        plt.subplot(4, 4, (idx-1)*4 + 2)
        sns.boxplot(y=df_processed[var], color='lightcoral')
        plt.title(f'{var} - Box Plot', fontsize=12, fontweight='bold')
        plt.ylabel(var)
        
#Violin Plot

        plt.subplot(4, 4, (idx-1)*4 + 3)
        sns.violinplot(y=df_processed[var], color='lightgreen')
        plt.title(f'{var} - Violin Plot', fontsize=12, fontweight='bold')
        plt.ylabel(var)

#Bar Plot (Frequency)
            
        plt.subplot(4, 4, (idx-1)*4 + 4)
        value_counts = df_processed[var].value_counts().sort_index()
        plt.bar(value_counts.index, value_counts.values, color='mediumpurple', alpha=0.7)
        plt.title(f'{var} - Frequency Distribution', fontsize=12, fontweight='bold')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        
plt.tight_layout()
plt.show()


#Bivariate Analysis

demographic_vars = ['Sex', 'Age', 'Education', 'Income']
available_vars = [var for var in demographic_vars if var in df_processed.columns]
print(f"\nAvailable demographic variables: {available_vars}")

#Correlations

corr_pearson = df_processed[available_vars].corr(method='pearson')
corr_spearman = df_processed[available_vars].corr(method='spearman')

print("\nPearson Correlation:")
print(corr_pearson.round(4))

print("\nSpearman Correlation:")
print(corr_spearman.round(4))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#Heatmaps

#Age by Sex & Education

if all(v in available_vars for v in ['Sex', 'Education', 'Age']):
    pivot_age = df_processed.pivot_table(values='Age',
                                         index='Education',
                                         columns='Sex',
                                         aggfunc='mean')
    
    sns.heatmap(pivot_age, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=2, ax=axes[0])
    axes[0].set_title('Mean Age by Sex & Education', fontweight='bold')
    axes[0].set_xlabel('Sex (0=Female, 1=Male)')
    
#Income by Sex & Education

if all(v in available_vars for v in ['Sex', 'Education', 'Income']):
    pivot_income = df_processed.pivot_table(values='Income',
                                            index='Education',
                                            columns='Sex',
                                            aggfunc='mean')
    
    sns.heatmap(pivot_income, annot=True, fmt='.1f', cmap='Greens',
                linewidths=2, ax=axes[1])
    axes[1].set_title('Mean Income by Sex & Education', fontweight='bold')
    axes[1].set_xlabel('Sex (0=Female, 1=Male)')

plt.tight_layout()
plt.savefig('bivariate_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

#Greatest Correlations

correlations = []
for i in range(len(available_vars)):
    for j in range(i+1, len(available_vars)):
        correlations.append({
            'Pair': f"{available_vars[i]} - {available_vars[j]}",
            'r': corr_pearson.iloc[i, j]
        })

correlations_df = pd.DataFrame(correlations).sort_values('r', 
     key=abs, 
     ascending=False)

for idx, row in correlations_df.head(3).iterrows():
    print(f"  {row['Pair']}: r = {row['r']:.3f}")

        
