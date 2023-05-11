import requests
from bs4 import BeautifulSoup
import pandas as pd


def main():
    url = 'https://www.espn.com/tennis/rankings'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'Table'})
    headers = [header.text for header in table.find_all('th')]

    # Add the 'Nationality' column header
    headers.insert(2, 'Nationality')

    rows = table.find_all('tr')[1:]  # Exclude the header row
    data = []
    for row in rows:
        row_data = [td.text for td in row.find_all('td')]

        # Find the player's nationality and insert it into the row_data list
        nationality_img = row.find('img', {'class': 'Image Logo Logo__sm'})
        if nationality_img and 'title' in nationality_img.attrs:
            nationality = nationality_img['title']
        else:
            nationality = ''
        row_data.insert(2, nationality)

        data.append(row_data)

    df = pd.DataFrame(data, columns=headers)
    df = df.drop(df.columns[1], axis=1)
    df.to_csv('./TennisGameStatsAndBets/tennis_rankings.csv', index=False)


if __name__ == '__main__':
    main()
