import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_extensions as de
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
from nltk.stem import PorterStemmer
st = PorterStemmer()

stop_words = set(stopwords.words('english'))

newer_stopwords = ["that", "go", "yeah", "uh", "oh", "let", "hey", "okay"]
new_stopwords_list = stop_words.union(newer_stopwords)

new_stop_words = set(
    [word for word in new_stopwords_list if word not in {'not'}])


# Load processed dataframes
with open('./assets/df_lines.data', 'rb') as filehandle:
    df_lines = pickle.load(filehandle)

with open('./assets/ep_names.data', 'rb') as filehandle:
    episode_guide = pickle.load(filehandle)

with open('./assets/link_quote.data', 'rb') as filehandle:
    df_char = pickle.load(filehandle)

with open('./assets/top_mean_values.data', 'rb') as filehandle:
    top_10_mean_values = pickle.load(filehandle)

with open('./assets/word_df.data', 'rb') as filehandle:
    word_df = pickle.load(filehandle)

with open('./assets/Multinomial_CVect_SMOTE.sav', 'rb') as filehandle:
    pipeline = pickle.load(filehandle)


def check_episode(ep_number, book):
    '''
    Helper function to return name of episode given the episode number and Book
    '''
    return episode_guide[(episode_guide['ep_number'] == ep_number)
                         & (episode_guide['Book'] == book)].iloc[0, 0]

def get_info(kind, character):
    if kind == 'link':
        test_str = df_char[df_char.char==character][kind].values[0]
        # slicing off after .png
        res = test_str[:test_str.index('.png') + len('.png')]
        return res

    else:
        return df_char[df_char.char==character][kind].values[0]

# Top 10 charcaters (in number of lines)
top_10_characters = df_lines['Character'].value_counts()[:10]
df_lines_top10 = df_lines[( df_lines['Character'].isin(tuple(top_10_characters.index)) )]
lines_by_episode = df_lines_top10.groupby('total_number')[['polarity','subjectivity','word_count', 'ep_number', 'Book']].mean().reset_index()
lines_by_episode['ep_name'] = lines_by_episode.apply(lambda d: check_episode(d['ep_number'], d['Book']), axis=1)

by_character_top10 = df_lines[(df_lines['Character'].isin(
    tuple(top_10_characters.index)))].groupby('Character')[[
        'polarity', 'subjectivity', 'word_count'
    ]].mean().reset_index()

# Preparing character list of dictionaries for dropdown options
character_options = []
for char in top_10_characters.index:
    my_dict = {}
    my_dict['label'] = str(char)
    my_dict['value'] = str(char)
    character_options.append(my_dict)
character_options = sorted(character_options, key = lambda k: k['label'])

# Preparing attribute list of dictionaries for dropdown options
attribute_options = []
for att in ('polarity', 'subjectivity', 'word_count'):
    my_dict = {}
    my_dict['label'] = str(att)
    my_dict['value'] = str(att)
    attribute_options.append(my_dict)
attribute_options = sorted(attribute_options, key = lambda k: k['label'])
#####################################################
polarity_text = "Return the polarity score as a float within the range [-1.0, 1.0]. -1.0 refers to a very negative text, and 1.0 a very positive one."
subjective_text = "Return the subjectivity score as a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
wc_text = "word frequency in this text."

## Creating info cards
polarity_card = dbc.Card(
    dbc.CardBody(
        [
            html.P(polarity_text),
        ]
    )
)
subjectivity_card = dbc.Card(
    dbc.CardBody(
        [
            html.P(subjective_text),
        ]
    )
)
wc_card = dbc.Card(
    dbc.CardBody(
        [
            html.P(wc_text),
        ]
    )
)
## Function to plot top episodes by mean attribute
def plot_episode_rank(df, attribute, top=10):
    '''
    * Plots a bar plot for the 'attribute' value of each episode in df.
    * 'df' has to have a column named 'atribute' and one named 'ep_name' for text entry
    * Value of 'top' determines how many episodes to rank
    '''
    fig = px.bar(df.sort_values(by=attribute, ascending=False)[:top].reset_index(),
             y = attribute,
             text='ep_name',
             color= px.colors.qualitative.Light24[:top])
    fig.update_layout(
        showlegend=False,
        title="Top 10 episodes by mean "+ attribute,
        xaxis_title="Rank",
        yaxis_title="Mean "+attribute
    )
    return fig

# Function to plot evolution of attribute across episodes
def plot_top10_evolution(character, attribute):
    fig = go.Figure()

    for char in [character]:

        fig.add_scatter(x = top_10_mean_values[char]['total_number'],
                        y = top_10_mean_values[char][attribute],
                        hovertext=list(zip(top_10_mean_values[char]['ep_number'],
                                      top_10_mean_values[char]['Book'])),
                        name = char)
    fig.update_layout(
        title="Evolution of "+ attribute,
        xaxis_title="Episode number (in entire series)",
        yaxis_title="Mean "+attribute+" of lines"
    )
    fig.update_layout(hovermode="x unified")

    fig.add_vrect(
        # Add background color for Book 1
        x0=1, x1=20,
        fillcolor="LightBlue", opacity=0.5,
        layer="below", line_width=0,
        annotation_text='Book 1', annotation_position="top left",
        annotation_font_color = "Blue", annotation_font_size=16
    )

    fig.add_vrect(
        # Add background color for Book 2
        x0=20, x1=40,
        fillcolor="LightGreen", opacity=0.5,
        layer="below", line_width=0,
        annotation_text='Book 2', annotation_position="top left",
        annotation_font_color = "DarkGreen", annotation_font_size=16
    )

    fig.add_vrect(
        # Add background color for Book 3
        x0=40, x1=64,
        fillcolor="Red", opacity=0.2,
        layer="below", line_width=0,
        annotation_text='Book 3', annotation_position="top left",
        annotation_font_color = "crimson", annotation_font_size=16
    )


    return fig

# Function to return most frequent words by character
def character_word_freq_plot(char, top=20):
    '''
    Plot top 'top' more frequent words for character 'char'
    '''
    freq = FreqDist(sum(df_lines[df_lines['Character']==char]['script_clean'].map(word_tokenize), []))
    freq_df = pd.DataFrame(list(freq.items()), columns = ["Word","Frequency"])
    freq_df = freq_df.sort_values(by='Frequency', ascending=False)
    fig = px.bar(freq_df[:top], x = 'Word', y = 'Frequency',
             color=px.colors.qualitative.Light24[:top])
    fig.update_layout(
        showlegend=False,
        title=f"Top {top} more frequent words for {char}",
        xaxis_title="Word",
        yaxis_title="Frequency"
    )
    return fig

# Function to predict which avatar text sounds like
def get_prediction(pipe, text):

    # Lower case:
    clean_text = text.lower()
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]','',clean_text)
    # Remove stop words
    clean_text = " ".join(s for s in word_tokenize(clean_text) if s not in new_stop_words)
    # Stemming
    clean_text = " ".join([st.stem(word) for word in word_tokenize(clean_text)])

    # Make prediction
    preds = pipe.predict_proba([clean_text])[0]

    # print(f'Probability of being an Aang line: {preds[0]}')
    # print(f'Probability of being a Korra line: {preds[1]}')

    # return (f'Probability of being an Aang line: {round(preds[0]*100, 2)}%\n', f'Probability of being a Korra line: {round(preds[1]*100, 2)}%')
    return (round(preds[0]*100, 2), round(preds[1]*100, 2))



##################### START OF APP
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
server = app.server

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

## Creating the sidebar for multipage handling
sidebar = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Card(
                            dbc.CardImg(src="https://i.pinimg.com/originals/b3/2a/a5/b32aa5e4d8e2070af0f5826c7c2c3302.png", top=True)
                        ),
                        html.H5("Exploring Character lines \n The Last Airbender",
                            style = {'textAlign':'center'}),

                        html.H6(["Made with ",
                            html.A('Dash', href = 'https://dash.plotly.com/',
                                target = '_blank)'),
                            html.Br(),
                            html.Br(),
                            "by ",
                            html.A('Bruno V. Ribeiro', href = 'https://github.com/BrunoBVR',
                                target = '_blank)'),
                            ], style={'textAlign':'right'}),
                    ]
                ),
            ],
            color="light",
            inverse=False,
            outline=False
        ),
        # html.H2("Sidebar", className="display-4"),
        html.Hr(),

        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Character data", href="/char-info", active = "exact"),
                dbc.NavLink("Text classifier", href="/classifier", active = "exact")            ],
            vertical = True,
            pills = True,
        ),
        dbc.Card(
            [
                dbc.CardBody([
                    html.H6(["",
                        html.A('GitHub for project',
                            href = 'https://github.com/BrunoBVR/projectATLA',
                            target = '_blank)')
                        ], style={'textAlign':'left'})
                ])
            ],
            color='warning'),
    ],
    style = SIDEBAR_STYLE,
)

## Creating content for each page
content = html.Div(id="page-content", children=[], style = CONTENT_STYLE)

## Main body of app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

## Callback for each different page
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
################################################## MAIN PAGE
    if pathname == "/":
        return[
            html.H1("General info on the data",
                style = {'textAlign':'center'}),
            html.Hr(),

            dcc.Graph(id='top_10_lines',
                figure = (px.bar(top_10_characters, color=px.colors.qualitative.Light24[:10]).update_layout(showlegend=False,
                          title="Top 10 Characters with more lines",
                          xaxis_title="Character",
                          yaxis_title="Number of lines"))
            ),
            html.Hr(),
            html.H2("Get to know the attributes we are exploring:",
                style = {'textAlign':'center'}),
            dbc.Row([
                dbc.Col([
                dbc.Button(
                    "Polarity",
                    id="pol-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    polarity_card,
                    id="pol_collapse",
                    is_open=False,
                )
                ], width = 4),
                dbc.Col([
                dbc.Button(
                    "Subjectivity",
                    id="sub-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    subjectivity_card,
                    id="sub_collapse",
                    is_open=False,
                )
                ], width = 4),
                dbc.Col([
                dbc.Button(
                    "Word Count",
                    id="wc-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    wc_card,
                    id="wc_collapse",
                    is_open=False,
                )
                ], width = 4),
            ]),
            html.Hr(),

            html.H2("Ranking episodes",
                style = {'textAlign':'left'}),

            dbc.Row([
                dbc.Col(
                    dcc.Slider(id='episode-slider',
                        min = 5,
                        max = 20,
                        step = 1,
                        marks = {i: str(i) for i in range(1,21)},
                        value = 5
                ), width = 6),
                dbc.Col(
                    dcc.Dropdown(id='attribute-choice',
                                options = attribute_options,
                                style={'color': '#000000'},
                                value = 'polarity',
                                placeholder = 'Select attribute.'
                                )
                ,width = 6)
            ]),
            dcc.Graph(id = 'episode_graph',
                figure = plot_episode_rank(lines_by_episode, 'polarity', top = 5)),

            html.Hr(),
            html.H2("Distribtuion of attributes",
                style = {'textAlign':'left'}),

            dbc.Row([
                dbc.Col(dcc.Graph(id='pol_box',
                    figure = (px.box(df_lines_top10, x="Character", y="polarity",
                                color='Character'))), width=6),
                dbc.Col(dcc.Graph(id='pol_mean',
                    figure = (px.bar(by_character_top10, x='Character', y='polarity',
                            title='Mean Polarity').update_xaxes(categoryorder="total descending"))), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sub_box',
                    figure = (px.box(df_lines_top10, x="Character", y="subjectivity",
                                color='Character'))), width=6),
                dbc.Col(dcc.Graph(id='sub_mean',
                    figure = (px.bar(by_character_top10, x='Character', y='subjectivity',
                            title='Mean Subjectivity').update_xaxes(categoryorder="total descending"))), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='wc_box',
                    figure = (px.box(df_lines_top10, x="Character", y="word_count",
                                color='Character'))), width=6),
                dbc.Col(dcc.Graph(id='wc_mean',
                    figure = (px.bar(by_character_top10, x='Character', y='word_count',
                            title='Mean Word Count').update_xaxes(categoryorder="total descending"))), width=6),
            ]),

        ]



################################################## Character page
    elif pathname == "/char-info":
        return[
            html.H1("Character information",
                style = {'textAlign':'center'}),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H3("Choose character to display information:") , width = 6),
                dbc.Col(
                    dcc.Dropdown(id='char-choice',
                                options = character_options,
                                style={'color': '#000000'},
                                value = 'Aang',
                                placeholder = 'Select character.'
                                ), width = 6),
            ]),

            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div(id='char-card'),
                    dbc.Alert(
                        html.H6(id='number-of-lines'),
                         color = 'info'),
                    dbc.Alert(
                        html.H6(id='word-diversity'),
                         color = 'warning'),
                ], width = 3),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(html.H5("Choose attribute to see evolution:",
                            style = {'textAlign':'right'})),
                        dbc.Col(
                            dcc.Dropdown(id='attribute-char-choice',
                                        options = attribute_options,
                                        style={'color': '#000000'},
                                        value = 'polarity',
                                        placeholder = 'Select attribute.'
                                        ),
                        )
                    ]),
                    dcc.Graph(id = 'line-plot',
                        figure = plot_top10_evolution('Aang', 'polarity')),
                    dbc.Alert(
                        html.H4(id='ep-name'),
                         color = 'primary')
                ], width = 9)

            ]),

            html.Hr(),
            html.H2('Word Frequency', style = {'textAlign':'center'}),
            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.Img(id = 'wordcloud', src=app.get_asset_url('Aang.png'), style={'height':'70%'})
                    ], width = 6),
                dbc.Col([
                    dcc.Slider(id='word-slider',
                        min = 5,
                        max = 20,
                        step = 1,
                        marks = {i: str(i) for i in range(1,21)},
                        value = 5
                    ),
                    dcc.Graph(id = 'freq-words')
                ], width = 6),
            ])

        ]

################################################## Classifier
    elif pathname == "/classifier":
        return[
            html.H1("Text Classifier",
                style = {'textAlign':'center'}),
            html.Hr(),

            html.H3("Which Avatar do you sound like?", style = {'textAlign':'center'}),
            html.Hr(),

            dbc.Row(
                [html.Div(id='b-a', children = de.BeforeAfter(before="assets/avatar1.gif", after="assets/avatar2.jpeg", width=340, height=213))],
                justify='center'
            ),
            dbc.Input(id="text-input", placeholder="Type something...", type="text"),
            dbc.Button('Submit text for prediction.', id='submit-text', color='primary', n_clicks=0),
            dbc.Row([
                dbc.Col(dbc.Alert(id='aang-prob', color='warning')),
                dbc.Col(dbc.Alert(id='korra-prob', color='info')),
            ]),
            dcc.Graph(id="a-k")

        ]


    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
##############################################################
############################### START OF INTERACTIVE CALLBACKS
##############################################################

############################# Main page
@app.callback(
    Output("pol_collapse", "is_open"),
    [Input("pol-button", "n_clicks")],
    [State("pol_collapse", "is_open")],
)
def toggle_pol_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("sub_collapse", "is_open"),
    [Input("sub-button", "n_clicks")],
    [State("sub_collapse", "is_open")],
)
def toggle_sub_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("wc_collapse", "is_open"),
    [Input("wc-button", "n_clicks")],
    [State("wc_collapse", "is_open")],
)
def toggle_wc_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("episode_graph", "figure"),
    Input("episode-slider", "value"),
    Input("attribute-choice", "value")
)
def get_ep_graph(eps, att):
    return plot_episode_rank(lines_by_episode, att, top = eps)

############################# Character page
@app.callback(
    Output("line-plot", "figure"),
    Input("attribute-char-choice", "value"),
    Input("char-choice", "value")
)
def get_evo_graph(att, char):
    return plot_top10_evolution(char, att)

@app.callback(
    Output("char-card", "children"),
    Output("number-of-lines", "children"),
    Output("word-diversity", "children"),
    Output("wordcloud", "src"),
    Input("char-choice", "value")
)
def get_char_card(char):
    card = (
        dbc.Card([
            dbc.CardImg(src=get_info('link', char), top=True),
            dbc.CardBody([
                html.P(get_info('quote', char))
            ])
        ], color='primary', inverse = True)
    )
    return card, f"Lines in show: {top_10_characters[char]}.",\
        f"Word diversity: {word_df[word_df.Character == char]['word_diversity'].iloc[0]}.",\
        app.get_asset_url(str(char)+'.png')

@app.callback(
    Output("ep-name", "children"),
    Input("line-plot", "clickData")
)
def get_click_episode(data):
    if not data:
        return "Click on episode"
    else:
        ep_book = eval(data['points'][0]['hovertext'])
        return "Episode name: "+check_episode(ep_book[0], ep_book[1])

@app.callback(
    Output("freq-words", "figure"),
    Input("char-choice", "value"),
    Input("word-slider","value")
)
def get_freq_words(char, num):
    return character_word_freq_plot(char, top=num)

############################# Classifier page
@app.callback(
    Output("aang-prob", "children"),
    Output("korra-prob", "children"),
    Output("a-k", "figure"),
    Input("submit-text", "n_clicks"),
    State("text-input", "value")
)
def make_prediction(n_clicks, text):
    if text is not None:
        aang, korra = get_prediction(pipeline, text)

        df = pd.DataFrame(zip(('Aang', 'Korra'), (aang, korra), (' ',' ')),
            columns=['character', 'probability', 'Predicted'])
        fig = px.bar(df, x="probability", y = 'Predicted', color='character', orientation='h',
            height=200)

        return f'Probability of being an Aang line: {aang}%\n', f'Probability of being a Korra line: {korra}%', fig
    else:
        df = pd.DataFrame(zip(('Aang', 'Korra'), (0.5, 0.5), (' ',' ')),
            columns=['character', 'probability', 'Predicted'])
        fig = px.bar(df, x="probability", y = 'Predicted', color='character', orientation='h',
            height=200)
        return "Aang's prediction here.", "Korra's prediction here.", fig

#####################################
if __name__ == '__main__':
    app.run_server()
