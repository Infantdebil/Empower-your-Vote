import os
import re
import csv
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from main import launch_ui
from retriever import Retriever
  
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key/big-cargo-399608-2f03cc8b576f.json'

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# YouTube API Client
youtube = build("youtube", "v3", developerKey=API_KEY)

# Data folder
DATA_FOLDER = "data/youtube"
os.makedirs(DATA_FOLDER, exist_ok=True)

def load_party_channels_from_csv():
    """Loads the party channels from the CSV file into a list of dictionaries."""

    # File path for storing party channels
    DATA_FOLDER = "data/youtube"
    CSV_FILE_PATH = os.path.join(DATA_FOLDER, "0_party_channels_overview.csv")

    if not os.path.exists(CSV_FILE_PATH):
        print(f"⚠️ No CSV file found at {CSV_FILE_PATH}. Returning default PARTY_CHANNELS.")
        return PARTY_CHANNELS  # Return default if no file exists
    
    with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        loaded_channels = [row for row in reader]  # Convert CSV rows to a list of dicts

    return loaded_channels

def get_channel_id(youtube_url):
    """Extracts YouTube channel ID from a username-based URL."""
    
    # Initialize retriever
    retriever = Retriever()

    username = re.search(r"@([\w-]+)$", youtube_url)
    if username:
        try:
            response = youtube.search().list(
                part="snippet",
                type="channel",
                q=username.group(1),
                maxResults=1
            ).execute()
            return response["items"][0]["id"]["channelId"] if response["items"] else None
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print("⚠️ YouTube API quota exceeded. Skipping channel lookup.")
                return launch_ui()
            print(f"⚠️ Error fetching channel ID: {e}")
            return None
    return None

def get_recent_videos(channel_id, since_date="2024-11-06T00:00:00Z"):
    """Fetches recent videos from a YouTube channel since a given date."""
    try:
        response = youtube.channels().list(id=channel_id, part="contentDetails").execute()
    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            print("⚠️ YouTube API quota exceeded. Skipping video retrieval.")
            return []
        print(f"⚠️ Error fetching channel details: {e}")
        return []

    playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_list = []
    next_page_token = None

    while True:
        try:
            playlist_response = youtube.playlistItems().list(
                playlistId=playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token
            ).execute()
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print("⚠️ YouTube API quota exceeded. Skipping remaining videos.")
                return video_list
            print(f"⚠️ Error fetching playlist videos: {e}")
            return video_list  # Stop fetching if there's an error

        for item in playlist_response["items"]:
            video_id = item["snippet"]["resourceId"]["videoId"]
            creation_date = item["snippet"]["publishedAt"]

            if creation_date >= since_date:
                video_list.append({
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "creation_date": creation_date,
                    "description": item["snippet"]["description"],
                    "author": item["snippet"]["channelTitle"],
                    "source": f"https://www.youtube.com/watch?v={video_id}"
                })

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    return video_list

def get_transcript(video_id):
    """Fetches the transcript of a YouTube video if available"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["de", "en"])
        return " ".join([entry["text"] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def load_existing_videos(party):
    """Loads processed videos from CSV, ensuring 'embedded' column exists."""
    file_path = f"{DATA_FOLDER}/processed_videos_{party}.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # ✅ Ensure "embedded" column always exists
        if "embedded" not in df.columns:
            df["embedded"] = True  # Default to True
        
        return df
    
    return pd.DataFrame(columns=["video_id", "title", "description", "author", "source", "embedded"])

def save_new_videos(party, videos, embedded_status=True):
    """Saves new videos to processed list with embedded status."""
    file_path = f"{DATA_FOLDER}/processed_videos_{party}.csv"
    df = load_existing_videos(party)

    # Standardize column names
    required_columns = ["video_id", "title", "description", "author", "source", "embedded"]
    df = df.reindex(columns=required_columns, fill_value=False)  # Fill missing columns

    # Ensure embedded status is correctly applied
    for video in videos:
        video["embedded"] = embedded_status  

    # Convert to DataFrame with matching columns
    new_videos_df = pd.DataFrame(videos).reindex(columns=required_columns, fill_value=False)

    # Concatenate with existing DataFrame
    df = pd.concat([df, new_videos_df], ignore_index=True)

    # Ensure no extra columns before saving
    df.to_csv(file_path, index=False)

def process_party_videos():
    """Processes new videos and saves them before embedding."""

    # Initialize party youtube channels
    PARTY_CHANNELS = load_party_channels_from_csv()

    for party_info in PARTY_CHANNELS:
        party, url = party_info["party"], party_info["URL"]
        print(f"🔎 Checking {party} YouTube channel...")

        # Checking for existing channel of each party
        channel_id = get_channel_id(url)
        if not channel_id:
            print(f"Could not retrieve channel ID for {party}")
            continue

        # Load already processed videos
        existing_videos = load_existing_videos(party)

        # Separate already embedded videos
        existing_urls = set(existing_videos[existing_videos["embedded"] == True]["source"])

        # Get new videos
        new_videos = get_recent_videos(channel_id)
        unprocessed_videos = [v for v in new_videos if v["source"] not in existing_urls]

        if not unprocessed_videos:
            print(f"✅ No new unprocessed videos for {party}. Skipping processing.")
            continue

        for video in unprocessed_videos:
            print(f"📌 Processing: {video['title']} ({video['source']})")

            transcript = get_transcript(video["video_id"])
            if transcript:
                video["transcript"] = transcript
                save_path = f"{DATA_FOLDER}/video_transcript_content_{party}.csv"
            else:
                save_path = f"{DATA_FOLDER}/videos_without_transcript_content_{party}.csv"

            pd.DataFrame([video]).to_csv(save_path, mode="a", header=not os.path.exists(save_path), index=True)

        # ✅ Save processed videos BEFORE embedding
        save_new_videos(party, unprocessed_videos, embedded_status=True)

        print(f"✅ Processed {len(unprocessed_videos)} new videos for {party}")

def load_yt_videos():
    """Loads video transcript files and prepares them for database."""

    PARTY_CHANNELS = load_party_channels_from_csv()

    # Preprocessing party youtube channel and video data
    process_party_videos()
    
    # Processing all videos from each youtube channel of each party
    video_documents = []
    print(f"Channel screening done.")

    for party_info in PARTY_CHANNELS:
        party = party_info["party"]
        file_path = f"{DATA_FOLDER}/video_transcript_content_{party}.csv"

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            video_documents.append({
                "metadata": {
                    "filename": row["title"],  # Video title as filename
                    "page_number": "0",  # Videos don't have pages, so set to "0"
                    "creation_date": row["creation_date"],  # Video upload date
                    "author": row["author"],  # YouTube channel name
                    "party": party,  # Party name
                    "description": row["description"],  # Video description
                    "source": row["source"],  # Video URL
                },
                "text": row.get("transcript", "[Transcript not available]")  # Handle missing transcripts
            })
        mark_videos_as_processed(video_documents["party"],video_documents["source"] )

    print(f"Loaded {len(video_documents)} YouTube video transcripts.")
    print(f"Loaded {video_documents} YouTube video transcripts.")

    # mark videos in processed file as done and got to the final chunking and embedding / needs to be corrected!

    return video_documents

def mark_videos_as_processed(party, video_sources):
    """Marks the given videos as successfully embedded in the CSV file."""
    file_path = f"{DATA_FOLDER}/processed_videos_{party}.csv"
    
    print(f"marking all preprocessed files as processed")

    if not os.path.exists(file_path):
        print(f"⚠️ No processed video file for {party}, skipping embedding update.")
        return
    
    df = pd.read_csv(file_path)
    
    # ✅ Ensure "embedded" column exists
    if "embedded" not in df.columns:
        df["embedded"] = True  

    # ✅ Update embedded status only for processed videos
    df.loc[df["source"].isin(video_sources), "embedded"] = True
    
    # ✅ Save the updated CSV
    df.to_csv(file_path, index=True)
    print(f"✅ Successfully marked {len(video_sources)} videos as embedded for {party}.")