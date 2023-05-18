import glob
import random

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.src.layers import Dropout
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

def load_and_preprocess_data(df):
    """Load and preprocess the dataset."""

    keep_cols = ['ATP', 'Location', 'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Winner',
                 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'Wsets', 'Lsets', 'MaxW', 'MaxL', 'AvgW', 'AvgL']
    df = df[keep_cols]
    df['Location'] = pd.factorize(df['Location'])[0]
    df['Tournament'] = pd.factorize(df['Tournament'])[0]
    df['Series'] = pd.factorize(df['Series'])[0]
    df['Court'] = pd.factorize(df['Court'])[0]
    df['Surface'] = pd.factorize(df['Surface'])[0]
    df['Round'] = pd.factorize(df['Round'])[0]
    df['RankDiff'] = df['WRank'] - df['LRank']
    df = df.fillna(0)
    return df

def get_training_data(df):
    """Generate training data."""
    player_dict = {}
    training = pd.DataFrame(
        columns=["P1", "P2", "P1 Rank", "P2 Rank", "Rank Diff",
                 "P1 VS P2",
                 "P2 VS P1", "Max P1 Odds", "Avg P1 Odds", "Max P2 Odds", "Avg P2 Odds", "Win Pct P1",
                 "Win Pct P2", "Cubed Rank Diff", "Winner"])
    id = 0
    for i in range(len(df)):

        row = df.iloc[i]

        date = row.Date
        winner = row.Winner
        loser = row.Loser
        wrank = row.WRank
        lrank = row.LRank
        MaxW = row.MaxW
        MaxL = row.MaxL
        AvgW = row.AvgW
        AvgL = row.AvgL

        if winner not in player_dict:
            player_dict.update(
                {winner: {"id": id, "Total_Wins": 0, "Total_Losses": 0, "Wins": {loser: 0}, "Losses": {loser: 0}}})
            id += 1
        elif loser not in player_dict[winner]['Wins']:
            player_dict[winner]['Wins'].update({loser: 0})
            player_dict[winner]['Losses'].update({loser: 0})

        if loser not in player_dict:
            player_dict.update(
                {loser: {"id": id, "Total_Wins": 0, "Total_Losses": 0, "Wins": {winner: 0}, "Losses": {winner: 0}}})
            id += 1
        elif winner not in player_dict[loser]['Losses']:
            player_dict[loser]['Wins'].update({winner: 0})
            player_dict[loser]['Losses'].update({winner: 0})

        loser_wins = player_dict[loser]['Total_Wins']
        loser_losses = player_dict[loser]['Total_Losses']
        winner_wins = player_dict[winner]['Total_Wins']
        winner_losses = player_dict[winner]['Total_Losses']

        winner_v_loser = player_dict[winner]['Wins'][loser]
        loser_v_winner = player_dict[loser]['Wins'][winner]

        if random.choice([True, False]):
            p1, p1_rank, p1_wins, p1_losses, p1_p2, max_p1, avg_p1 = player_dict[winner][
                'id'], wrank, winner_wins, winner_losses, winner_v_loser, MaxW, AvgW
            p2, p2_rank, p2_wins, p2_losses, p2_p1, max_p2, avg_p2 = player_dict[loser][
                'id'], lrank, loser_wins, loser_losses, loser_v_winner, MaxL, AvgL
        else:
            p1, p1_rank, p1_wins, p1_losses, p1_p2, max_p1, avg_p1 = player_dict[loser][
                'id'], lrank, loser_wins, loser_losses, loser_v_winner, MaxL, AvgL
            p2, p2_rank, p2_wins, p2_losses, p2_p1, max_p2, avg_p2 = player_dict[winner][
                'id'], wrank, winner_wins, winner_losses, winner_v_loser, MaxW, AvgW

        rank_diff = p1_rank - p2_rank
        cubed_rank_diff = (p1_rank - p2_rank) ** 3

        p1_pw = .5
        if p1_wins + p1_losses > 0:
            p1_pw = p1_wins / (p1_wins + p1_losses)
        p2_pw = .5
        if p2_wins + p2_losses > 0:
            p2_pw = p2_wins / (p2_wins + p2_losses)

        training.loc[i] = [p1, p2, p1_rank, p2_rank, rank_diff, p1_p2, p2_p1,
                           max_p1, avg_p1, max_p2, avg_p2, p1_pw, p2_pw, cubed_rank_diff, player_dict[winner]['id']]
        player_dict[winner]['Total_Wins'] += 1
        if loser not in player_dict[winner]['Wins']:
            player_dict[winner]['Wins'].update({loser: 1})
            player_dict[winner]['Losses'].update({loser: 0})
        else:
            player_dict[winner]['Wins'][loser] += 1
        player_dict[loser]['Total_Losses'] += 1
        if winner not in player_dict[loser]['Losses']:
            player_dict[loser]['Wins'].update({winner: 0})
            player_dict[loser]['Losses'].update({winner: 1})
        else:
            player_dict[loser]['Losses'][winner] += 1
    training = training.astype(float)
    training['Winner'] = training.apply(lambda row: 1 if row['Winner'] == row['P1'] else 0, axis=1)

    training.drop(['P1', 'P2'], axis=1, inplace=True)

    return training

def plot_correlation_matrix(data):
    """Plot a correlation matrix for the data."""
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

def split_and_normalize_data(data):
    """Split data into training and testing sets and normalize features."""
    X = data.drop(columns='Winner')
    y = data['Winner']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def create_and_train_model(X_train, y_train):
    """Create and train a machine learning model."""
    # Define model architecture
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance on a test set."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

def analyze_predictions(model, X_test, y_test):
    """Analyze the model's predictions."""
    predictions = model.predict(X_test)
    # Convert predictions to binary outcomes
    binary_predictions = (predictions > 0.5).astype(int).flatten()

    # Get the model's confidence
    confidence = np.where(binary_predictions == 1, predictions.flatten(), 1 - predictions.flatten())

    # Create a DataFrame to compare the actual and predicted outcomes
    comparison_df = pd.DataFrame({
        'Outcome': y_test,
        'Prediction': binary_predictions,
        'Confidence': confidence
    })

    return comparison_df

def plot_profit_and_loss_vs_bet_confidence(df):
    """Plot profit and loss vs. bet confidence."""
    values = []
    number_of_bets = []
    confidences = [i / 100 for i in range(50, 100)]
    x_vals = [i for i in range(50, 100)]
    for i in confidences:
        df = df[df['Confidence'] > i]
        number_of_bets.append(len(df))
        df['value'] = df.apply(lambda row: -1000 if row['Outcome'] != row['Prediction'] else (row['Avg P2 Odds'] if row[
                                                                                                                        'Prediction'] == 0 else
                                                                                              row[
                                                                                                  'Avg P1 Odds']) * 1000 - 1000,
                               axis=1)

        values.append((df['value'].sum()))
        print(df)
        percentage = (df['Prediction'] == df['Outcome']).mean() * 100
        print(f"The percentage where Prediction equals Outcome is: {percentage}%")

    fig, axs = plt.subplots(2)

    print(values)
    axs[0].set_title('Profit and Loss Vs Bet Confidence')

    axs[0].plot(x_vals, values)
    axs[0].set(ylabel="Strategy Value ($)")
    axs[0].grid(True)  # Add grid to the first subplot

    axs[1].plot(x_vals, number_of_bets)
    axs[1].set(xlabel="Minimum Confidence Threshold (%)", ylabel="Number of Bets")
    axs[1].grid(True)  # Add grid to the second subplot

    plt.tight_layout()
    plt.show()

def vary_bet(df):
    """Vary bet based on confidence and analyze the result."""
    values = []
    number_of_bets = []
    confidences = [i / 100 for i in range(50, 100)]
    x_vals = [i for i in range(50, 100)]
    bet = 10000
    for i in confidences:

        bet = 10000 * (.9*i**20)

        print(bet)
        df = df[df['Confidence'] > i]
        number_of_bets.append(len(df))
        df['value'] = df.apply(lambda row: -bet if row['Outcome'] != row['Prediction'] else (row['Avg P2 Odds'] if row[
                                                                                                                       'Prediction'] == 0 else
                                                                                             row[
                                                                                                 'Avg P1 Odds']) * bet - bet,
                               axis=1)
        values.append((df['value'].sum()))
        print(df)
        percentage = (df['Prediction'] == df['Outcome']).mean() * 100
        print(f"The percentage where Prediction equals Outcome is: {percentage}%")

    fig, axs = plt.subplots(2)

    print(values)
    axs[0].set_title('Profit and Loss Vs Bet Confidence (Varying Bet Amount)')

    axs[0].plot(x_vals, values)
    axs[0].set(ylabel="Strategy Value ($)")
    axs[0].grid(True)  # Add grid to the first subplot

    axs[1].plot(x_vals, number_of_bets)
    axs[1].set(xlabel="Minimum Confidence Threshold (%)", ylabel="Number of Bets")
    axs[1].grid(True)  # Add grid to the second subplot

    plt.tight_layout()
    plt.show()

def main():
    """Main function to tie all steps together."""
    path = r'data'  # use your path
    all_files = glob.glob(path + "/*.csv")

    list_of_dfs = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list_of_dfs.append(df)

    # Concatenate all data into one DataFrame
    big_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    df = load_and_preprocess_data(big_df)
    data = get_training_data(df)
    print(data.tail().to_string())
    plot_correlation_matrix(data)
    X_train, X_test, y_train, y_test = split_and_normalize_data(data)
    model = create_and_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    comparison_df = analyze_predictions(model, X_test, y_test)
    with_odds = comparison_df.join(data[['Avg P1 Odds', 'Avg P2 Odds', "P1 Rank", "P2 Rank", "Rank Diff", "Win Pct P1", "Win Pct P2"]])
    print(with_odds)
    with_odds.to_csv('with_odds.csv', index=False)
    #plot_profit_and_loss_vs_bet_confidence(with_odds)
    #vary_bet(with_odds)

if __name__ == "__main__":
    main()

