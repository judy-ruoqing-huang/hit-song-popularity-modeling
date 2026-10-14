import requests
from bs4 import BeautifulSoup
import csv
import time
import logging
import re
from unidecode import unidecode
import os

logging.basicConfig(level=logging.INFO)

def clean_rank(rank):
    # Clean rank by removing text within parentheses
    clean_rank = re.sub(r'\(.*?\)', '', rank).strip()
    return clean_rank

def clean_title(title):
    # Remove double quotes and trim spaces
    cleaned_title = title.replace('"', '').strip()
    return cleaned_title

def clean_artist(artist):
    # Split by common separators and clean each part
    separators = ['featuring', 'feat.', ',', '&', 'and']
    for sep in separators:
        if sep in artist.lower():
            artist = re.split(re.escape(sep), artist, flags=re.IGNORECASE)[0].strip()

    # Normalize the artist name to convert non-English characters to closest English equivalents
    artist = unidecode(artist)

    return artist

def fetch_wikipedia_data(start_year, end_year, output_csv):
    header = ["Year", "Rank", "Title", "Artist"]

    # Write the header to the CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for year in range(start_year, end_year + 1):
        url = f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
        logging.info(f"Fetching data for year {year} from URL: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad response status
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data for year {year}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table:
            billboard_data = []
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip the header row
                cells = row.find_all('td')
                if len(cells) >= 3:
                    rank = clean_rank(cells[0].get_text(strip=True))
                    title = clean_title(cells[1].get_text(strip=True))
                    artist_full = cells[2].get_text(strip=True)

                    # Clean artist
                    cleaned_artist = clean_artist(artist_full)

                    billboard_data.append([year, rank, title, cleaned_artist])

            # Write the current year's data to the CSV file
            with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(billboard_data)
        else:
            logging.warning(f"No table found on the page for year {year}")

        time.sleep(1)  # Add a delay to be kind to the server

    logging.info(f"Billboard Hot 100 data saved to: {output_csv}")

if __name__ == "__main__":
    start_year = 1946  # Start year for scraping
    end_year = 2023  # End year for scraping

    # Directory path
    directory = 'BillboardExplain/data'
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Full file path
    output_csv = os.path.join(directory, 'wikipedia_hot_100_history.csv')

    fetch_wikipedia_data(start_year, end_year, output_csv)
