import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotify API credentials
CLIENT_ID = 'd2040dec092f4369ab7c7b63727de61a'  # Replace with your client ID
CLIENT_SECRET = '1cf373b758564f7d9511b4664309beb1'  # Replace with your client secret

# Authenticate with Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define the track title and artist
title = 'Big Shit Poppin'
artist = 'T.I.'

# Create the search query
query = f"track:{title} artist:{artist}"

# Perform the search
result = spotify.search(q=query, type='track', limit=1)

# Display the result
if result['tracks']['items']:
    track = result['tracks']['items'][0]
    print(f"Track found: {track['name']} by {track['artists'][0]['name']}")
    print(f"Track ID: {track['id']}")
    print(f"Track URL: {track['external_urls']['spotify']}")
else:
    print("Track not found.")
