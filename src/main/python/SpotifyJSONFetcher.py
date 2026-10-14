import os
import pandas as pd
import requests
from pandas import json_normalize
from tqdm import tqdm
from datetime import datetime
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directory path
directory = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'data')
os.makedirs(directory, exist_ok=True)

# Read Spotify API credentials from a CSV file
credentials_file_path = os.path.join(directory, 'spotify_credentials.csv')
credentials_df = pd.read_csv(credentials_file_path)
CREDENTIALS = credentials_df.to_dict(orient='records')

# Setup central logging (file only, no terminal)
logging.basicConfig(filename=os.path.join(directory, 'process_log.log'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Function to clean the text
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

# Function to get the access token
def get_access_token(credentials):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': credentials['client_id'],
        'client_secret': credentials['client_secret'],
    })
    auth_response_data = auth_response.json()
    return auth_response_data['access_token']

# Function to get track IDs from Spotify API
def get_track_ids(track_name, artist_name, access_token):
    search_url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'q': f'track:{track_name} artist:{artist_name}',
        'type': 'track',
        'limit': 1
    }
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['tracks']['items']:
            track_id = data['tracks']['items'][0]['id']
            return track_id
    return None

# Function to get detailed track data using batch 'GET /tracks' and 'GET /audio-features'
def get_batch_track_data(track_ids, access_token):
    tracks_url = 'https://api.spotify.com/v1/tracks'
    audio_features_url = 'https://api.spotify.com/v1/audio-features'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        tracks_response = requests.get(tracks_url, headers=headers, params={'ids': ','.join(track_ids)})
        audio_features_response = requests.get(audio_features_url, headers=headers, params={'ids': ','.join(track_ids)})

        if tracks_response.status_code == 200 and audio_features_response.status_code == 200:
            tracks_data = tracks_response.json()['tracks']
            audio_features_data = audio_features_response.json()['audio_features']
            return tracks_data, audio_features_data

        elif tracks_response.status_code == 429 or audio_features_response.status_code == 429:
            retry_after = int(tracks_response.headers.get('Retry-After', audio_features_response.headers.get('Retry-After', 1)))
            logger.warning(f'429 Too Many Requests encountered. Retrying after {retry_after} seconds.')
            time.sleep(retry_after)
            retry_count += 1
        else:
            return None, None

    return None, None

# Function to get audio analysis data using 'GET /audio-analysis/{id}'
def get_audio_analysis(track_id, access_token):
    audio_analysis_url = f'https://api.spotify.com/v1/audio-analysis/{track_id}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        response = requests.get(audio_analysis_url, headers=headers)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 1))
            logger.warning(f'429 Too Many Requests encountered for track {track_id}. Retrying after {retry_after} seconds.')
            time.sleep(retry_after)
            retry_count += 1
        else:
            return None
    return None

# Function to process a group of track_ids with a specific access token
def process_track_group(track_ids_group, access_token, group_index):
    # Create directory for logs and results for this group
    group_directory = os.path.join(directory, f'group_{group_index}')
    os.makedirs(group_directory, exist_ok=True)

    # Set up logging to a file
    log_file_path = os.path.join(group_directory, f'group_{group_index}_log.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Read existing results if available
    existing_results_file = os.path.join(group_directory, f'group_{group_index}_results.csv')
    existing_results = pd.read_csv(existing_results_file) if os.path.exists(existing_results_file) else None

    # Initialize a list to hold batch flattened dataframes
    batch_flattened_dfs = []

    # Define batch size
    batch_size = 50

    # Iterate through track_ids_group in batches
    for i in tqdm(range(0, len(track_ids_group), batch_size), desc=f"Group {group_index}: Fetching Track Data"):
        batch_track_ids = track_ids_group[i:i + batch_size]
        retry_count = 0
        while retry_count < 3:
            try:
                track_ids_batch = [track_id[0] for track_id in batch_track_ids]

                # Check existing results for current batch
                if existing_results is not None:
                    existing_batch_data = existing_results[existing_results['track_id'].isin(track_ids_batch)]
                    existing_batch_data = existing_batch_data[['track_id', 'year', 'rank']]

                    # Merge existing data with batch_track_ids to get missing track_ids
                    existing_track_ids = set(existing_batch_data['track_id'].values)
                    missing_track_ids = [track_id for track_id in batch_track_ids if track_id[0] not in existing_track_ids]

                    # Append existing data to batch_flattened_dfs
                    batch_flattened_dfs.append(existing_batch_data)

                    # Process missing_track_ids from Spotify
                    track_ids_batch = missing_track_ids

                # Fetch track data from Spotify API for missing track IDs
                tracks_data, audio_features_data = get_batch_track_data(track_ids_batch, access_token)
                all_flattened_data = []  # Initialize the batch list here

                # Process each track in the batch
                for track_info in batch_track_ids:
                    track_id = track_info[0]
                    year = track_info[1]
                    rank = track_info[2]

                    # Initialize data with basic information
                    track_data = {
                        'track_id': track_id,
                        'year': year,
                        'rank': rank
                    }

                    # Fetch and add tracks_data if available
                    if tracks_data:
                        for track in tracks_data:
                            if track['id'] == track_id:
                                track_data.update(track)
                                break

                    # Fetch and add audio_features_data if available
                    if audio_features_data:
                        for features in audio_features_data:
                            if features['id'] == track_id:
                                track_data['audio_features'] = features
                                break

                    # Fetch and add audio_analysis_data if available
                    if 'audio_features' in track_data:  # Only fetch audio analysis if audio features are found
                        audio_analysis_data = get_audio_analysis(track_id, access_token)
                        if audio_analysis_data:
                            track_data['audio_analysis'] = audio_analysis_data

                    # Add to all_flattened_data if all necessary data is available
                    if all(key in track_data for key in ['id', 'audio_features', 'audio_analysis']):
                        all_flattened_data.append(track_data)

                # Convert all_flattened_data to DataFrame in a single step
                batch_flattened_df = json_normalize(all_flattened_data)

                # Save batch DataFrame to CSV
                if i == 0 and existing_results is None:  # Check if it's the first batch and no existing results
                    batch_flattened_df.to_csv(existing_results_file, mode='a', header=True, index=False)
                else:
                    batch_flattened_df.to_csv(existing_results_file, mode='a', header=False, index=False)

                # Append to list of batch dataframes
                batch_flattened_dfs.append(batch_flattened_df)

                break  # Exit retry loop on success

            except requests.exceptions.RequestException as e:
                if e.response.status_code in [401, 403]:
                    current_credential_index = (current_credential_index + 1) % len(CREDENTIALS)
                    access_token = get_access_token(CREDENTIALS[current_credential_index])
                elif e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 1))
                    time.sleep(retry_after)
                    current_credential_index = (current_credential_index + 1) % len(CREDENTIALS)
                    access_token = get_access_token(CREDENTIALS[current_credential_index])
                else:
                    logger.error(f"Unexpected error occurred: {e}")
                    retry_count += 1
                logger.error(f"Error: {e.response.status_code} for batch starting at index {i}")
                retry_count += 1

    # Concatenate all batch results into a single DataFrame for the group
    final_df = pd.concat(batch_flattened_dfs, ignore_index=True)

    # Save final_df to CSV
    final_df.to_csv(existing_results_file, index=False)

    # Remove the file handler after processing
    logger.removeHandler(file_handler)
    file_handler.close()

    return final_df

# Full file path (moved outside the function)
input_csv_file_path = os.path.join(directory, 'wikipedia_hot_100_history.csv')

# Read the input CSV file outside the function
tracks_df = pd.read_csv(input_csv_file_path)

# Initialize variables for rotating credentials
current_credential_index = 0

# CSV file path for storing found track IDs
tracks_csv_file = os.path.join(directory, 'tracks_found.csv')

# Check if the CSV file exists
missed_tracks = []  # Initialize missed_tracks here
if os.path.exists(tracks_csv_file):
    # Read track IDs from CSV file
    track_ids_df = pd.read_csv(tracks_csv_file)
    track_ids = list(zip(track_ids_df['track_id'], track_ids_df['year'], track_ids_df['rank']))
    print("Track IDs read from CSV file:")
    print(track_ids)
else:
    # Fetch track IDs from Spotify API if CSV file doesn't exist
    track_ids = []

    for index, row in tqdm(tracks_df.iterrows(), total=tracks_df.shape[0], desc="Fetching Track IDs"):
        # Apply the cleaning function to titles and artists
        track_name = clean_text(row['Title'])
        artist_name = clean_text(row['Artist'])

        track_id = get_track_ids(track_name, artist_name, CREDENTIALS[current_credential_index])
        if track_id:
            track_ids.append((track_id, row['Year'], row['Rank']))  # Include Year and Rank here
        else:
            missed_tracks.append(row)
        time.sleep(0.05)  # Small delay to prevent rate limiting

    # Save found tracks to CSV for future runs
    df_found_tracks = pd.DataFrame(track_ids, columns=['track_id', 'year', 'rank'])
    df_found_tracks.to_csv(tracks_csv_file, index=False)

# Divide track_ids into 5 groups
num_groups = 5
track_ids_groups = [track_ids[i::num_groups] for i in range(num_groups)]

# Use ThreadPoolExecutor to process each group in parallel
with ThreadPoolExecutor(max_workers=num_groups) as executor:
    futures = []
    for i, track_ids_group in enumerate(track_ids_groups):
        access_token = get_access_token(CREDENTIALS[i % len(CREDENTIALS)])
        futures.append(executor.submit(process_track_group, track_ids_group, access_token, i))

    # Combine results from all groups (if needed)
    all_results = []
    for future in as_completed(futures):
        group_result = future.result()
        all_results.append(group_result)

# Optionally, combine all_results if needed

# Save final combined results (if applicable)
# final_df.to_csv(output_csv_file_path, index=False)

print(f"Processing completed successfully.")
print(f"Check individual group directories for logs and results.")
