from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled,RequestBlocked,VideoUnavailable

#First get a transcript of a youtube video

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id , languages=["en"])

        transcript = ' '.join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video")
        return None
    except RequestBlocked:
        print("YouTube blocked the request. Try again later or use another network.\n")
        return None
    except VideoUnavailable:
        print("Video is unavaliable")
        return None