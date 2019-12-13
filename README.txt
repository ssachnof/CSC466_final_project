please compile with python3.6 or higher
Samuel Sachnoff(ssachnof@calpoly.edu)
The data cleaning folder contains the scripts we used in order to clean the Carnegie play by play dataset we used. Additionally,
all of the datafiles that we used in our models are in the data folder. All of our model generation is in the models folder.
Please feel free to use any of the datasets that we have created in the data folder.

Questions:
Given a team’s first half offensive stats, can we predict who wins the game?
Given a team’s whole game stats, can we predict who wins the game?

In order to run either of the above 2 questions, please run predictor_win_game.py. This file will run the cross validation
for either of the random forest or decision tree classifiers that we created. Additionally, this file takes a couple of different
command line arguments.

if you would like to run the cross validation for a given model:
    python3.6 predictor_win_game.py <filepath> <method> <hyperparameter_value> <selected columns>
    filepath: string representing the path to the input datafile(must either be all_games.csv or all_games_half1.csv)
    method: -rf for random forest or -dt for decision tree
    hyperparameter value:
        if using random forest classifier:
            integer representing the number of trees to create in the ensemble classifier
        if using decision tree:
            float represent the information gain threshold
    selected columns:
        a list of space delmited string representing the columns to use while building the model(ie. a b c d where a,b,c,d are different
        columns)
if you would like to run the model selection for either random forest or decision tree(I strongly suggest that you don't due to
the significant runtime)
    python3.6 predictor_win_game.py <filepath> <method> <max_cols>
    filepath- path to data file
    method: -rf or -dt for random forest or decision tree respectively
    max_cols: integer representing the maximum number of columns to use while performing model selection

