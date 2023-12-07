from dash import html as html
from dash import dcc as dcc
import pandas as pd
import plotly.express as px

import numpy as np
from plotly.subplots import make_subplots
import plotly.subplots
import plotly.graph_objects as go
from scipy import stats


def create_main(df):

    # Figure 1 Question : Which Disease has the highest datavalue in our dataset?
    df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')
    disease_data = df.groupby('topic')['datavalue'].sum().reset_index()
    max_disease = disease_data.loc[disease_data['datavalue'].idxmax()]

    fig1 = px.bar(disease_data, x='topic', y='datavalue', title='Total Datavalue by Disease',
             labels={'topic': 'Disease', 'datavalue': 'Total Datavalue'})
    fig1.update_traces(marker_color=['rgba(0, 0, 255, 0.7)' if topic == max_disease['topic']
                                     else 'rgba(0, 0, 0, 0.1)' for topic in disease_data['topic']])
    fig1.update_layout(xaxis_title='Disease', yaxis_title='Total Datavalue')

    # Figure 2 Question: Finding the Most common data source used for reporting of all diseases using grouped bar chart
    df_chronic_diseases = df[df['topic'].isin([
        'Cardiovascular Disease', 'Alcohol', 'Arthritis', 'Asthma', 'Cancer',
        'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease',
        'Mental Health', 'Tobacco', 'Overarching Conditions', 'Oral Health',
        'Reproductive Health', 'Diabetes', 'Immunization',
        'Nutrition, Physical Activity, and Weight Status', 'Disability',
        'Older Adults'
    ])]
    disease_data_by_source = df_chronic_diseases.groupby(['topic', 'datasource'])['datavalue'].sum().reset_index()

    fig2 = px.bar(disease_data_by_source,
                  x='topic',
                  y='datavalue',
                  color='datasource',
                  title='Count of Cases for Each Chronic Disease Grouped by Data Source',
                  labels={'datavalue': 'Cases', 'topic': 'Chronic Disease'},
                  category_orders={"topic": [
                      'Cardiovascular Disease', 'Alcohol', 'Arthritis', 'Asthma', 'Cancer',
                      'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease',
                      'Mental Health', 'Tobacco', 'Overarching Conditions', 'Oral Health',
                      'Reproductive Health', 'Diabetes', 'Immunization',
                      'Nutrition, Physical Activity, and Weight Status', 'Disability',
                      'Older Adults'
                  ]}
                  )
    fig2.update_layout(xaxis_title='Chronic Disease', yaxis_title='Cases')

    # Figure 3 Question: Disease ups and downs analysis over the years
    # List of diseases
    diseases = [
        'Cardiovascular Disease',
        'Alcohol',
        'Arthritis',
        'Asthma',
        'Cancer',
        'Chronic Kidney Disease',
        'Chronic Obstructive Pulmonary Disease',
        'Mental Health',
        'Tobacco',
        'Overarching Conditions',
        'Oral Health',
        'Reproductive Health',
        'Diabetes',
        'Immunization',
        'Nutrition, Physical Activity, and Weight Status',
        'Disability',
        'Older Adults',
    ]
    # Initialize an empty dictionary to store disease counts
    disease_counts = {}
    for disease in diseases:
        disease_data = df[df['topic'] == disease]
        year_counts = disease_data.groupby('yearstart').size().reset_index(name='count')
        disease_counts[disease] = year_counts

    fig3 = make_subplots(rows=len(diseases), cols=1, subplot_titles=diseases, shared_xaxes=True)
    for i, disease in enumerate(diseases):
        fig3.add_trace(go.Bar(x=disease_counts[disease]['yearstart'], y=disease_counts[disease]['count'],
                         name=disease), row=i+1, col=1)
    fig3.update_layout(height=1500, width=800, title_text="Disease Occurrences Over the Years",
                  xaxis_title="Year", yaxis_title="Count")

    # Figure 4 Question: analysis of chronic diseases with ethinicity
    # Filter data for diseases and include only 'Male' and 'Female' in 'stratification1'
    df_filtered = df[df['topic'].isin(['Cardiovascular Disease', 'Alcohol', 'Arthritis', 'Asthma', 'Cancer',
                                       'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease',
                                       'Mental Health', 'Tobacco', 'Overarching Conditions', 'Oral Health',
                                       'Reproductive Health', 'Diabetes', 'Immunization',
                                       'Nutrition, Physical Activity, and Weight Status', 'Disability',
                                       'Older Adults'])]
    df_filtered = df_filtered[df_filtered['stratification1'].isin(['Male', 'Female'])]

    fig4 = px.box(df_filtered, x='stratification1', y='datavalue', color='topic',
             title='Distribution of Disease Counts by Ethnicity',
             labels={'stratification1': 'Gender', 'datavalue': 'Count', 'topic': 'Disease'},
             category_orders={'topic': sorted(df_filtered['topic'].unique())},
             height=600)
    fig4.update_layout(xaxis_title='Gender', yaxis_title='Count')

    # Figure 5 Question :What is the comparison of Cancer and Chronic Obstructive Pulmonary Diseases (Based on the Count)
    df_cancer_copd = df[df['topic'].isin(['Cancer', 'Chronic Obstructive Pulmonary Disease'])]
    disease_counts = df_cancer_copd.groupby('topic').size().reset_index(name='count')

    fig5 = px.bar(disease_counts,
             x='topic',
             y='count',
             title='Comparison of Cancer and COPD Occurrences',
             labels={'count': 'Count', 'topic': 'Disease'},
             color='topic',
             color_discrete_sequence=px.colors.qualitative.Set2)

    # Figure 6 Question: Time series chart depicting the incidence of cancer over the yearstart
    cancer_data = df[df['topic'] == 'Cancer']
    cancer_time_series = cancer_data.groupby('yearstart')['datavalue'].sum().reset_index()

    fig6 = px.line(cancer_time_series, x='yearstart', y='datavalue',
              title='Incidence of Cancer Over the Years',
              labels={'yearstart': 'Year', 'datavalue': 'Cancer Incidence'},
              markers=True)
    fig6.update_layout(xaxis_title='Year', yaxis_title='Cancer Incidence')

    # Figure 7 Question: Analysing which locations have the highest and lowest incidences of cancer
    df_cancer = df[(df['topic'] == 'Cancer') & (df['locationabbr'] != 'US')]
    cancer_incidence_by_location = df_cancer.groupby('locationabbr')['datavalue'].sum().reset_index()

    fig7 = px.choropleth(cancer_incidence_by_location,
                         locations='locationabbr',
                         locationmode='USA-states',
                         color='datavalue',
                         scope="usa",
                         title='Cancer Incidence by Location (Choropleth Map)',
                         labels={'datavalue': 'Cancer Cases', 'locationabbr': 'Location'},
                         color_continuous_scale='Viridis')
    fig7.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)'))

    # Figure 8 Question: identifying states with unusually high or low counts of cancer cases - states that contains outliers
    df_filtered = df[df['locationdesc'] != 'United States']
    sum_by_state = df_filtered.groupby('locationdesc')['datavalue'].sum().reset_index(name='sum_datavalue')
    # Calculate Z-scores for each state
    sum_by_state['z_score'] = stats.zscore(sum_by_state['sum_datavalue'])
    # Set Z-score threshold
    z_threshold = 2
    # Identify outliers based on the Z-scores
    sum_by_state['outlier'] = sum_by_state['z_score'] > z_threshold

    # Plot Z-scores against states with a threshold line using Plotly Express
    fig8 = px.bar(sum_by_state, x='locationdesc', y='z_score',
                  title='Z-Scores for Each State (Excluding United States)',
                  labels={'locationdesc': 'State', 'z_score': 'Z-Score'},
                  color='outlier',  # Color by the 'outlier' column
                  color_discrete_map={False: 'blue', True: 'red'},
                  text='sum_datavalue')  # Display sum_datavalue as text on bars
    fig8.add_shape(
        dict(
            type='line',
            yref='y',
            y0=z_threshold,
            y1=z_threshold,
            xref='paper',
            x0=0,
            x1=1,
            line=dict(color='red', dash='dash'),
        )
    )
    fig8.update_layout(xaxis_title='State', yaxis_title='Z-Score',
                       xaxis=dict(tickangle=90))

    # Figure 9 Question: Time series depicting the trends observed in the distribution of cancer cases among the different races
    cancer_data = df[df['topic'] == 'Cancer']
    other_genders_time_series = cancer_data[~cancer_data['stratification1'].isin(['Male', 'Female', 'Overall'])]
    other_genders_time_series = other_genders_time_series.groupby(['stratification1', 'yearstart'])[
        'datavalue'].sum().reset_index()

    fig9 = px.line(other_genders_time_series, x='yearstart', y='datavalue', color='stratification1',
                   title='Trends in the Distribution of Cancer Among different races',
                   labels={'yearstart': 'Year', 'datavalue': 'Cancer Cases', 'stratification1': 'race'},
                   markers=True)
    fig9.update_layout(xaxis_title='Year', yaxis_title='Cancer Cases')

    # Figure 10 Question: Distribution of cancer by Ethnicity using pie chart
    df_cancer_stratification = df[
        (df['topic'] == 'Cancer') & ~df['stratification1'].isin(['Male', 'Female', 'Overall'])]
    stratification_distribution = df_cancer_stratification.groupby('stratification1').size().reset_index(name='count')

    fig10 = px.pie(stratification_distribution,
                 names='stratification1',
                 values='count',
                 title='Distribution of Cancer Cases by Stratification (Excluding Male, Female, and Overall)',
                 labels={'count': 'Count', 'stratification1': 'Stratification'},
                 color='stratification1',
                 color_discrete_sequence=px.colors.qualitative.Set3)

    # Figure 11 Distribution of Cancer disease in all the states in comparison to the Gender
    # Filter data for cancer cases
    cancer_data = df[df['topic'] == 'Cancer']
    cancer_data = cancer_data[cancer_data['locationdesc'] != 'United States']
    gender_location_distribution = cancer_data[cancer_data['stratification1'].isin(['Male', 'Female'])]
    gender_location_distribution = gender_location_distribution.groupby(['stratification1', 'locationdesc'])[
        'datavalue'].sum().reset_index()

    # Plot the bar graph
    fig11 = px.bar(gender_location_distribution, x='locationdesc', y='datavalue', color='stratification1',
                 title='Distribution of Cancer Cases by Gender and Location (Excluding US)',
                 labels={'locationdesc': 'Location', 'datavalue': 'Cancer Cases', 'stratification1': 'Gender'},
                 category_orders={'locationdesc': sorted(gender_location_distribution['locationdesc'].unique())},
                 barmode='group')

    # Customize layout
    fig11.update_layout(xaxis_title='Location', yaxis_title='Cancer Cases')

    # Figure 12
    cancer_data = df[df['topic'] == 'Cancer']
    gender_time_series = cancer_data[cancer_data['stratification1'].isin(['Male', 'Female'])]
    gender_time_series = gender_time_series.groupby(['stratification1', 'yearstart'])['datavalue'].sum().reset_index()

    # Time series depicting the trends observed in the distribution of cancer cases within the different Gender.
    # Plot the time series chart
    fig12 = px.line(gender_time_series, x='yearstart', y='datavalue', color='stratification1',
                    title='Trends in the Distribution of Cancer Cases by Gender Over Time',
                    labels={'yearstart': 'Year', 'datavalue': 'Cancer Cases', 'stratification1': 'Gender'},
                    markers=True)

    # Customize layout
    fig12.update_layout(xaxis_title='Year', yaxis_title='Cancer Cases')

    # Figure 13
    # Create a new column 'keyword' based on certain questions so as to group common questions and analyse the data
    df['keyword'] = df['question'].map({
        'Invasive cancer of the oral cavity or pharynx, incidence': 'oral cancer',
        'Cancer of the oral cavity and pharynx, mortality': 'oral cancer',
        'Invasive cancer of the prostate, incidence': 'prostate cancer',
        'Cancer of the prostate, mortality': 'prostate cancer',
        'Invasive cancer (all sites combined), incidence': 'invasive cancer',
        'Invasive cancer (all sites combined), mortality': 'invasive cancer',
        'Invasive cancer of the female breast, incidence': 'breast cancer',
        'Melanoma, mortality': 'skin cancer',
        'Cancer of the female breast, mortality': 'breast cancer',
        'Invasive cancer of the cervix, incidence': 'cervix cancer',
        'Cancer of the female cervix, mortality': 'cervix cancer',
        'Cancer of the colon and rectum (colorectal), incidence': 'colon cancer',
        'Cancer of the colon and rectum (colorectal), mortality': 'colon cancer',
        'Cancer of the lung and bronchus, incidence': 'lung cancer',
        'Cancer of the lung and bronchus, mortality': 'lung cancer',
        'Invasive melanoma, incidence': 'skin cancer'
    })
    cancer_data = df[df['keyword'].notnull()]
    cancer_types_time_series = cancer_data.groupby(['keyword', 'yearstart'])['datavalue'].sum().reset_index()
    #show the trend of the types of cancer over the years

    # Plot the time series chart
    fig13 = px.line(cancer_types_time_series, x='yearstart', y='datavalue', color='keyword',
                  title='Trends in the Distribution of Cancer Cases by Cancer Types Over Time',
                  labels={'yearstart': 'Year', 'datavalue': 'Cancer Cases', 'keyword': 'Cancer Type'},
                  markers=True)
    fig13.update_layout(xaxis_title='Year', yaxis_title='Cancer Cases')

    # Figure 14
    # Create a new column 'keyword' based on certain questions so as to group common questions and analyse the data
    df['keyword'] = df['question'].map({
        'Invasive cancer of the oral cavity or pharynx, incidence': 'oral cancer',
        'Cancer of the oral cavity and pharynx, mortality': 'oral cancer',
        'Invasive cancer of the prostate, incidence': 'prostate cancer',
        'Cancer of the prostate, mortality': 'prostate cancer',
        'Invasive cancer (all sites combined), incidence': 'invasive cancer',
        'Invasive cancer (all sites combined), mortality': 'invasive cancer',
        'Invasive cancer of the female breast, incidence': 'breast cancer',
        'Melanoma, mortality': 'skin cancer',
        'Cancer of the female breast, mortality': 'breast cancer',
        'Invasive cancer of the cervix, incidence': 'cervix cancer',
        'Cancer of the female cervix, mortality': 'cervix cancer',
        'Cancer of the colon and rectum (colorectal), incidence': 'colon cancer',
        'Cancer of the colon and rectum (colorectal), mortality': 'colon cancer',
        'Cancer of the lung and bronchus, incidence': 'lung cancer',
        'Cancer of the lung and bronchus, mortality': 'lung cancer',
        'Invasive melanoma, incidence': 'skin cancer'
    })
    # Filter data for California
    california_cancer_data = df[(df['topic'] == 'Cancer') & (df['locationdesc'] == 'California')]
    combined_cancer_distribution = california_cancer_data.groupby('keyword')['datavalue'].sum().reset_index()

    # Since California has more Cancer topic, Which Cancer type is more common in California

    # Plot the pie chart
    fig14 = px.pie(combined_cancer_distribution,
                 names='keyword',
                 values='datavalue',
                 title='Distribution of Cancer Cases in California by Combined Cancer Type',
                 labels={'datavalue': 'Cancer Cases', 'keyword': 'Combined Cancer Type'},
                 hover_name='keyword')

    # Figure 15
    cancer_data = df[df['topic'] == 'Cancer']
    cancer_data = cancer_data[cancer_data['locationdesc'] != 'United States']
    data_source_distribution = cancer_data.groupby(['datasource', 'locationdesc'])['datavalue'].sum().reset_index()

    # analyzing what kind of data source is most commonly is used in the all the states to report cancer cases using a TreeMap
    # treemap
    fig15 = px.treemap(data_source_distribution,
                     path=['datasource', 'locationdesc'],
                     values='datavalue',
                     title='Distribution of Cancer Cases Reporting Data Source Across Different States',
                     labels={'datavalue': 'Cancer Cases'},
                     color='datavalue')

    # Figure 16
    df_mortality = df[(df['topic'] == 'Cancer') & (df['locationabbr'] != 'US')]
    mortality_questions = df_mortality[df_mortality['question'].str.contains('mortality', case=False, regex=True)]
    # Cancer mortality rates analysis with the locations using box plot

    # box plot
    fig16 = px.box(mortality_questions,
                 x='locationabbr',
                 y='datavalue',
                 title='Cancer Mortality Rates by Location (Excluding US)',
                 labels={'datavalue': 'Mortality Rate', 'locationabbr': 'Location'},
                 color='locationabbr',
                 color_discrete_sequence=px.colors.qualitative.Set3)  # Using a different color sequence

    # Customize layout
    fig16.update_layout(xaxis_title='Location', yaxis_title='Mortality Rate')

    # Figure 17
    # Group questions based on keywords
    keyword_mapping = {
        'oral cancer': ['Invasive cancer of the oral cavity or pharynx, incidence',
                        'Cancer of the oral cavity and pharynx, mortality'],
        'prostate cancer': ['Invasive cancer of the prostate, incidence', 'Cancer of the prostate, mortality'],
        'invasive cancer': ['Invasive cancer (all sites combined), incidence',
                            'Invasive cancer (all sites combined), mortality'],
        'invasive breast cancer': ['Invasive cancer of the female breast, incidence'],
        'skin cancer': ['Melanoma, mortality', 'Invasive melanoma, incidence'],
        'breast cancer': ['Cancer of the female breast, mortality'],
        'cervix cancer': ['Invasive cancer of the cervix, incidence', 'Cancer of the female cervix, mortality'],
        'colon cancer': ['Cancer of the colon and rectum (colorectal), incidence',
                         'Cancer of the colon and rectum (colorectal), mortality'],
        'lung cancer': ['Cancer of the lung and bronchus, incidence', 'Cancer of the lung and bronchus, mortality']
    }
    # Create a new column 'keyword' based on grouping
    df['keyword'] = df['question'].apply(lambda x: next((k for k, v in keyword_mapping.items() if x in v), None))
    df_female_cancer = df[(df['keyword'].notnull()) & (df['stratification1'] == 'Female')]
    female_cancer_distribution = df_female_cancer.groupby('keyword')['datavalue'].sum().reset_index()
    # Which type of cancer is most among females? Assumption is breast cancer

    # Create a pie chart
    fig17 = px.pie(female_cancer_distribution,
                 names='keyword',
                 values='datavalue',
                 title='Distribution of Cancer Types Among Females',
                 labels={'datavalue': 'Incidence Rate', 'keyword': 'Cancer Type'},
                 color='keyword',
                 color_discrete_sequence=px.colors.qualitative.Set3)

    # Figure 18
    # Group questions based on keywords
    keyword_mapping = {
        'oral cancer': ['Invasive cancer of the oral cavity or pharynx, incidence',
                        'Cancer of the oral cavity and pharynx, mortality'],
        'prostate cancer': ['Invasive cancer of the prostate, incidence', 'Cancer of the prostate, mortality'],
        'invasive cancer': ['Invasive cancer (all sites combined), incidence',
                            'Invasive cancer (all sites combined), mortality'],
        'invasive breast cancer': ['Invasive cancer of the female breast, incidence'],
        'skin cancer': ['Melanoma, mortality', 'Invasive melanoma, incidence'],
        'breast cancer': ['Cancer of the female breast, mortality'],
        'cervix cancer': ['Invasive cancer of the cervix, incidence', 'Cancer of the female cervix, mortality'],
        'colon cancer': ['Cancer of the colon and rectum (colorectal), incidence',
                         'Cancer of the colon and rectum (colorectal), mortality'],
        'lung cancer': ['Cancer of the lung and bronchus, incidence', 'Cancer of the lung and bronchus, mortality']
    }
    # Create a new column 'keyword' based on grouping
    df['keyword'] = df['question'].apply(lambda x: next((k for k, v in keyword_mapping.items() if x in v), None))
    df_male_cancer = df[(df['keyword'].notnull()) & (df['stratification1'] == 'Male')]
    male_cancer_distribution = df_male_cancer.groupby('keyword')['datavalue'].sum().reset_index()
    # Analysing Which type of cancer is most among males. Assumption is prostrate cancer

    # Create a pie chart
    fig18 = px.pie(male_cancer_distribution,
                 names='keyword',
                 values='datavalue',
                 title='Distribution of Cancer Types Among Males',
                 labels={'datavalue': 'Incidence Rate', 'keyword': 'Cancer Type'},
                 color='keyword',
                 color_discrete_sequence=px.colors.qualitative.Set3)

    # Figure 19
    # Group questions based on keywords
    keyword_mapping = {
        'oral cancer': ['Invasive cancer of the oral cavity or pharynx, incidence',
                        'Cancer of the oral cavity and pharynx, mortality'],
        'prostate cancer': ['Invasive cancer of the prostate, incidence', 'Cancer of the prostate, mortality'],
        'invasive cancer': ['Invasive cancer (all sites combined), incidence',
                            'Invasive cancer (all sites combined), mortality'],
        'invasive breast cancer': ['Invasive cancer of the female breast, incidence'],
        'skin cancer': ['Melanoma, mortality', 'Invasive melanoma, incidence'],
        'breast cancer': ['Cancer of the female breast, mortality'],
        'cervix cancer': ['Invasive cancer of the cervix, incidence', 'Cancer of the female cervix, mortality'],
        'colon cancer': ['Cancer of the colon and rectum (colorectal), incidence',
                         'Cancer of the colon and rectum (colorectal), mortality'],
        'lung cancer': ['Cancer of the lung and bronchus, incidence', 'Cancer of the lung and bronchus, mortality']
    }
    # Create a new column 'keyword' based on grouping
    df['keyword'] = df['question'].apply(lambda x: next((k for k, v in keyword_mapping.items() if x in v), None))
    df_other_stratifications = df[
        (df['keyword'].notnull()) & ~df['stratification1'].isin(['Male', 'Female', 'Overall'])]
    other_stratification_distribution = df_other_stratifications.groupby('keyword')['datavalue'].sum().reset_index()
    # Type of cancer common among the different races

    # Create a pie chart
    fig19 = px.pie(other_stratification_distribution,
                 names='keyword',
                 values='datavalue',
                 title='Distribution of Cancer Types (Other Stratifications)',
                 labels={'datavalue': 'Incidence Rate', 'keyword': 'Cancer Type'},
                 color='keyword',
                 color_discrete_sequence=px.colors.qualitative.Set3)

    # Figure 20
    df_cancer = df[df['topic'] == 'Cancer']
    df_melted = df_melted = pd.melt(
        df_cancer,
        id_vars=['yearstart'],
        value_vars=['lowconfidencelimit', 'highconfidencelimit'],
        var_name='confidence_interval',
        value_name='confidence_level'
    )
    # Analysis of cancertopic confidence level in the dataset

    # Create a box plot for confidence levels
    fig20 = px.box(
        df_melted, x='yearstart', y='confidence_level', color='confidence_interval',
        labels={'confidence_level': 'Confidence Level', 'confidence_interval': 'Confidence Interval'},
        title='Distribution of Confidence Levels Across Topic - Cancer'
    )

    # Update layout
    fig20.update_layout(width=1000, height=500)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20


if __name__ == '__main__':
    pass