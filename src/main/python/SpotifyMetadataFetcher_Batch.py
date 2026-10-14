import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
import time
import csv
from datetime import datetime
import re
import requests
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Spotify API credentials
CLIENT_ID = 'd2040dec092f4369ab7c7b63727de61a'  # Replace with your client ID
CLIENT_SECRET = '1cf373b758564f7d9511b4664309beb1'  # Replace with your client secret

def authenticate_spotify():
    try:
        credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        spotify = spotipy.Spotify(client_credentials_manager=credentials_manager, requests_timeout=10)
        logging.info("Successfully authenticated with Spotify.")
        return spotify
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        raise

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[-/:+.]', ' ', text)
    text = text.strip()
    text = re.sub(r'[“”‘’]', "'", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def fetch_additional_metadata(tracks, spotify, retries=5):
    track_ids = [track['id'] for track in tracks]
    metadata_list = []
    
    for attempt in range(retries):
        try:
            audio_features = spotify.audio_features(tracks=track_ids)
            break
        except (spotipy.SpotifyException, requests.exceptions.ReadTimeout) as e:
            if attempt < retries - 1:
                if isinstance(e, spotipy.SpotifyException) and e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 1))
                    logging.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    logging.warning(f"Error fetching audio features (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"Maximum retries reached. Skipping {len(track_ids)} tracks.")
                audio_features = [None] * len(track_ids)  # Ensure audio_features is a list of Nones if retries exhausted

    for i, track in enumerate(tracks):
        artist_id = track['artist_id']
        artist = None
        for attempt in range(retries):
            try:
                artist = spotify.artist(artist_id)
                break
            except (spotipy.SpotifyException, requests.exceptions.ReadTimeout) as e:
                if attempt < retries - 1:
                    if isinstance(e, spotipy.SpotifyException) and e.http_status == 429:
                        retry_after = int(e.headers.get('Retry-After', 1))
                        logging.warning(f"Rate limit hit while fetching artist. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                    else:
                        logging.warning(f"Error fetching artist details (attempt {attempt + 1}): {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Maximum retries reached for artist {artist_id}.")
                    artist = {'genres': [], 'followers': {'total': 0}}
        
        album = track['album']['name']
        popularity = track['popularity']
        duration_ms = track['duration_ms']
        
        genres = ', '.join(artist.get('genres', []))
        followers = artist.get('followers', {}).get('total', 0)
        
        features = audio_features[i] if audio_features[i] else {}
        
        try:
            analysis = spotify.audio_analysis(track['id'])
        except spotipy.SpotifyException as e:
            if e.http_status == 404:
                logging.warning(f"Audio analysis not found for track {track['Title']} by {track['Artist']}")
                analysis = None
            else:
                raise e
        
        metadata = {
            "Year": track['Year'],
            "Rank": track['Rank'],
            "Title": track['Title'],
            "Artist": track['Artist'],
            "Album": album,
            "Popularity (0-100)": popularity,
            "Duration (ms)": duration_ms,
            "Genres": genres,
            "Followers": followers,
            "Loudness (dB)": features.get('loudness', 0),
            "Tempo (BPM)": features.get('tempo', 0),
            "Key": features.get('key', 0),
            "Mode": features.get('mode', 0),
            "Time Signature": features.get('time_signature', 0),
            "Acousticness": features.get('acousticness', 0),
            "Danceability": features.get('danceability', 0),
            "Energy": features.get('energy', 0),
            "Instrumentalness": features.get('instrumentalness', 0),
            "Liveness": features.get('liveness', 0),
            "Speechiness": features.get('speechiness', 0),
            "Valence": features.get('valence', 0),
            "Analysis": analysis
        }
        
        logging.info(metadata)
        metadata_list.append(metadata)
        
        if not features:
            logging.warning(f"Audio features not found for track {track['Title']} by {track['Artist']}")
    
    return metadata_list

def search_track_with_retries(query, spotify, retries=5):
    for attempt in range(retries):
        try:
            result = spotify.search(q=query, type='track', limit=1)
            return result
        except (spotipy.SpotifyException, requests.exceptions.ReadTimeout) as e:
            if attempt < retries - 1:
                if isinstance(e, spotipy.SpotifyException) and e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 1))
                    logging.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    logging.warning(f"Error searching track with query '{query}' (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
            else:
                logging.error(f"Maximum retries reached for search query: '{query}'. Skipping.")
                return None
    return None

def main():
    # Directory path
    directory = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'data')
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Full file path
    input_csv_file_path = os.path.join(directory, 'wikipedia_hot_100_history.csv')
    # input_csv_file_path = os.path.join(directory, 'tracks_not_found_20240629_162020.csv')
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Output file paths with timestamp
    output_csv_file_path = os.path.join(directory, f'spotify_metadata_{timestamp}.csv')
    not_found_file_path = os.path.join(directory, f'tracks_not_found_{timestamp}.csv')

    # Authenticate with Spotify
    spotify = authenticate_spotify()

    # Read the input CSV file
    df = pd.read_csv(input_csv_file_path)

    # Prepare CSV writers
    with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as output_file, \
         open(not_found_file_path, mode='w', newline='', encoding='utf-8') as not_found_file:
        
        output_writer = csv.writer(output_file)
        not_found_writer = csv.writer(not_found_file)
        
        # Write headers
        output_writer.writerow(["Year", "Rank", "Title", "Artist", "Album", "Popularity (0-100)", "Duration (ms)", 
                               "Genres", "Followers", "Loudness (dB)", "Tempo (BPM)", "Key", "Mode", 
                               "Time Signature", "Acousticness", "Danceability", "Energy", 
                               "Instrumentalness", "Liveness", "Speechiness", "Valence", "Analysis"])
        
        not_found_writer.writerow(["Year", "Rank", "Title", "Artist"])

        batch_size = 100  # Maximum batch size for Spotify API search
        track_ids = []
        metadata_dict = {}

        for index, row in df.iterrows():
            year = row['Year']
            rank = row['Rank']
            title = row['Title']
            artist = row['Artist']

            # Apply the cleaning function to titles and artists
            cleaned_title = clean_text(title)
            cleaned_artist = clean_text(artist)

            query = f"track:{cleaned_title} artist:{cleaned_artist}"
            result = search_track_with_retries(query, spotify)  # Use the retry search function
            
            if result and result['tracks']['items']:
                track = result['tracks']['items'][0]
                track_id = track['id']
                track_data = {
                    "Year": year,
                    "Rank": rank,
                    "Title": title,
                    "Artist": artist,
                    "id": track_id,  # Store track ID for fetching audio features later
                    "album": track['album'],
                    "popularity": track['popularity'],
                    "duration_ms": track['duration_ms'],
                    "artist_id": track['artists'][0]['id']
                }
                track_ids.append(track_data)

                # Progress indicator
                logging.info(f"Processed {index + 1}/{len(df)}")

                # Process in batches
                if len(track_ids) >= batch_size:
                    metadata_batch = fetch_additional_metadata(track_ids, spotify)
                    for metadata in metadata_batch:
                        if metadata:
                            output_writer.writerow([
                                metadata['Year'], metadata['Rank'], metadata['Title'], metadata['Artist'],
                                metadata['Album'], metadata['Popularity (0-100)'], metadata['Duration (ms)'],
                                metadata['Genres'], metadata['Followers'], metadata['Loudness (dB)'],
                                metadata['Tempo (BPM)'], metadata['Key'], metadata['Mode'],
                                metadata['Time Signature'], metadata['Acousticness'], metadata['Danceability'],
                                metadata['Energy'], metadata['Instrumentalness'], metadata['Liveness'],
                                metadata['Speechiness'], metadata['Valence'], metadata['Analysis']
                            ])
                            logging.info(f"Track found and saved: {metadata['Title']} by {metadata['Artist']}")
                    track_ids = []
            else:
                logging.warning(f"No track found for: {title} by {artist}")
                not_found_writer.writerow([year, rank, title, artist])

        # Process remaining tracks if any
        if track_ids:
            metadata_batch = fetch_additional_metadata(track_ids, spotify)
            for metadata in metadata_batch:
                if metadata:
                    output_writer.writerow([
                        metadata['Year'], metadata['Rank'], metadata['Title'], metadata['Artist'],
                        metadata['Album'], metadata['Popularity (0-100)'], metadata['Duration (ms)'],
                        metadata['Genres'], metadata['Followers'], metadata['Loudness (dB)'],
                        metadata['Tempo (BPM)'], metadata['Key'], metadata['Mode'],
                        metadata['Time Signature'], metadata['Acousticness'], metadata['Danceability'],
                        metadata['Energy'], metadata['Instrumentalness'], metadata['Liveness'],
                        metadata['Speechiness'], metadata['Valence'], metadata['Analysis']
                    ])
                    logging.info(f"Track found and saved: {metadata['Title']} by {metadata['Artist']}")

if __name__ == "__main__":
    main()
