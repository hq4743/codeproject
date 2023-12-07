import dash
import dash_bootstrap_components as dbc
from dash import html as html
from dash import Input, Output
from dash import dcc as dcc
import pandas as pd
import plotly.express as px
import numpy as np

import pickle
from sklearn.metrics import mean_squared_error, r2_score
import createPlots
import dataCleaning
from ML_analysis import get_splitData


def get_split(data):
    pass

# Get the cleaned data
chronic_indicator_data = dataCleaning.main()

# Create different plots
fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20 = createPlots.create_main(
    chronic_indicator_data)

# Split data frame for model prediction
XTrain, XTest, labelsTrain, labelsTest = get_splitData(chronic_indicator_data)

# print("!!!!!!!!!!!!!!!!!", labelsTrain.shape, labelsTest.shape)
# Load Pre-trained model. These models were trained by us separately using the ML_analysis.py file
with open('final_dt_model.pkl', 'rb') as dtFile:
    final_dtModel = pickle.load(dtFile)
with open('final_rf_model.pkl', 'rb') as rfFile:
    final_rfModel = pickle.load(rfFile)

# Use the Decision Tree model to predict and compute MSE, and R^2 values
# Make predictions on the test set
dt_y_pred = final_dtModel.predict(XTest)

# Evaluate the model
dt_mse = mean_squared_error(labelsTest, dt_y_pred)
dt_r_squared = r2_score(labelsTest, dt_y_pred)

# print(f'Mean Squared Error: {dt_mse:.2f}')
# print(f'R-squared: {dt_mse:.2f}')

# Use the Random Forest model to predict and compute MSE, and R^2 values
# Make predictions on the test set
rf_y_pred = final_rfModel.predict(XTest)

# Evaluate the model
rf_mse = mean_squared_error(labelsTest, rf_y_pred)
rf_r_squared = r2_score(labelsTest, rf_y_pred)
# print("###############", labelsTest.shape, rf_y_pred.shape)

dt_visualization = pd.DataFrame({'y_test' : labelsTest, 'prediction' : dt_y_pred })

# Visualize the data
fig21 = px.scatter(dt_visualization, x='y_test', y='prediction', trendline='ols')
fig21.update_traces(marker=dict(color='green'))
fig21.update_xaxes(title='Test Labels')
fig21.update_yaxes(title='Predicted Labels')


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Define app layout
app.layout = dbc.Container([
    dbc.Row([

        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label='Demographics', value='demographic_analysis'),
                dcc.Tab(label='Diseases Analysis', value='diseases_analysis'),
                dcc.Tab(label='Cancer Data Analysis', value='cancer_analysis'),
                dcc.Tab(label='ML Analysis', value='ml')
            ], id='tabs', value='table'),
            html.Div(id='display-page'),
        ], md=12)
    ])
])



# Clustering:
# Tab 1: Demographic info -1,4,8,9,17,18,19,20
#
# Tab 2 :  Cancer type data : 5,6,7,12, 13,14
# Tab 3 :   Disease info - 2,3,10,11,15,16
# Tab 4: ML Data visualization


# Define callback to update display page
@app.callback(
    Output('display-page', 'children'),
    [
        Input('tabs', 'value')
    ]
)
def display_page(tab):
    if tab == 'demographic_analysis':
        layout = html.Div([
            html.H5('Total Datavalue by Disease'),
            html.P('The bar graphs show the data value for different Chronic diseases in the US. Each bar represents the data value associated with specific chronic diseases. “Chronic Obstructive Pulmonary Disease” has the highest data value, followed by cancer and cardiovascular disease  '),
            html.Div([
                dcc.Graph(id='fig1', figure=fig1)
            ]),
            html.H5(' Chronic Disease'),
            html.P('the figure shows the box plot for all the Chronic diseases occurrence in Male and Female. The prevalence of Chronic Obstructive Pulmonary Disease is notably high in women and the plot also indicates that cancer has a slightly higher occurrence in male as compared to women. '),

            html.Div([
                dcc.Graph(id='fig4', figure=fig4)
            ]),
            html.H5('Z-score Threshold'),
            html.P('Z-score graph helps us know the number of cases per capita based on the population of each state. From this graph, we see that Florida has higher cases of cancer. This graph is useful mainly to analyze a sudden surge in the number of cases for a particular location. '),

            html.Div([
                dcc.Graph(id='fig8', figure=fig8)
            ]),
            html.H5('Distribution of Cancer'),
            html.P(
                'This graph investigates the cancer cases across the ethnicity over some time. Initially, there were more cases among the White non-Hispanic group but over time this has flatlined. '),

            html.Div([
                dcc.Graph(id='fig9', figure=fig9)
            ]),
            html.H5('Cancer Composition Among Females'),
            html.P(
                'Among the female, we have more cases of breast cancer  and followed by the lung cancer'),

            html.Div([
                dcc.Graph(id='fig17', figure=fig17)
            ]),
            html.H5('Cancer Composition Among Males'),
            html.P(
                'Among the male we have more case of prostrate and followed by the lung cancer cancer cases '),

            html.Div([
                dcc.Graph(id='fig18', figure=fig18)
            ]),
            html.H5('Cancer Among Ethnicity'),
            html.P(
                'The White Non Hispanic has  prevailing cancer  for all cancer types '),

            html.Div([
                dcc.Graph(id='fig19', figure=fig19)
            ]),
            html.H5('Confidence Interval'),
            html.P(
                'Analysing the quality of the dataset based on interval values'),

            html.Div([
                dcc.Graph(id='fig20', figure=fig20)
            ]),
        ])
        return layout
    elif tab == "diseases_analysis":
        layout = html.Div([
            html.H5('Comparison Of Diseases'),
            html.P(
                'This interactive bar graph compares Cancer and Chronic Obstructive Pulmonary Disease counts, where Cancer has the highest count '),

            html.Div([
                dcc.Graph(id='fig5', figure=fig5)
            ]),
            html.H5('Time Series'),
            html.P(
                'This graph shows the number of new cases of Cancer over the years.'
                'There was a dip in the number of new cases in the year 2015'),

            html.Div([
                dcc.Graph(id='fig6', figure=fig6)
            ]),
            html.H5('Geographic Explanation'),
            html.P(
                'The choropleth graph explores which states have high or low rates of cancer based on raw data.'
                'Based on this we can see the highest is California followed by NY and Florida.'),

            html.Div([
                dcc.Graph(id='fig7', figure=fig7)
            ]),
            html.H5('Time Series Of Cancer Among Males'),
            html.P(
                ' We also did a time series evaluation to understand further about this. '
                'If you look closely initially the cases were more among males and gradually as time passed the number of cases in females started to increase.'),

            html.Div([
                dcc.Graph(id='fig12', figure=fig12)
            ]), \
            html.H5('Cancer Trend'),
            html.P(
                'Graph 13 shows the the trend of the types of cancer over the years: This is the prevailing tendency of different cancer types by cases between  2008 and 2015. It was observed that invasive cancer is the most predominant over the years considered.'),

            html.Div([
                dcc.Graph(id='fig13', figure=fig13)
            ]),
            html.H5('The Golden State'),
            html.P(
                'In one of our previous exploration (specifically figure 8), new York , florida and California are the state with the highest number of cancer cases. Given this information, we, further explored  the cases in one of these states (California),  there are a significant number of invasive cancer cases, however  breast cancer followed by lung cancer seems to be the most predominant'),

            html.Div([
                dcc.Graph(id='fig14', figure=fig14)
            ])
        ])
        return layout
    elif tab == 'cancer_analysis':
        layout = html.Div([
            html.H5('Datasource'),
            html.P(
                '"NVSS" datasource has reported highest "cardiovascular disease" and "CMS" and "SEDD" datasource has reported highest "Chronic Obstructive Pulmonary Disease" disease '),

            html.Div([
                dcc.Graph(id='fig2', figure=fig2)
            ]),
            html.H5('Chronic Diseases'),
            html.P(
                'Deep Dive into how these Diseases count changed over the years. The bar graphs show the variation based on disease count. It is safe to say that Cancer started during the year 2008. '),

            html.Div([
                dcc.Graph(id='fig3', figure=fig3)
            ]),
            html.H5('Composition of Cancer'),
            html.P(
                'This is another way to look at the no. of cases based on Ethnicity and this pie chart explores the frequency of cancer cases based on the data values for the entire dataset.'),

            html.Div([
                dcc.Graph(id='fig10', figure=fig10)
            ]),
            html.H5('Gender Competition'),
            html.P(
                'Graph shows the distribution of cancer cases among males and females across every state, and we can see that Female cancer cases are higher in most states. '),

            html.Div([
                dcc.Graph(id='fig11', figure=fig11)
            ]),
            html.H5('Common Datasource'),
            html.P(
                'Explains the different datasource in different locations using tree map to find the maximum cancer-detected locations'),

            html.Div([
                dcc.Graph(id='fig15', figure=fig15)
            ]),
            html.H5('Morality rates'),
            html.P(
                'This Graph shows that New York, Texas, California and Florida have high cases of mortality compared to any other state within the period'),

            html.Div([
                dcc.Graph(id='fig16', figure=fig16)
            ])
        ])
        return layout
    elif tab == 'ml':
        layout = html.Div([              
                html.H4('Mean Square Error of Models:'),
                
                html.P(f'Mean Square Error for Decision Tree: {dt_mse}'),
                html.P(f'Mean Square Error for Random Forest: {rf_mse}'),
                
                html.H4('R Square Score of Models:'),
                
                html.P(f'R Square score for Decision Tree: {dt_r_squared}'),
                html.P(f'R Square score for Random Forest: {rf_r_squared}'),

                html.H5('Prediction Visualization'),
                html.P('Since the MSE for Decision Tree is lower, we are visualizing the residuals plot for it.'),
                html.Div(
                    [
                        dcc.Graph(id='fig21', figure=fig21)
                    ]
                ),
                
                html.P('Explanation')
                ])
        return layout

if __name__ == '__main__':
    app.run_server(port=8888)
