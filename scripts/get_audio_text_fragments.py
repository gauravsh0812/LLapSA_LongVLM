import os
import json
import whisper
from moviepy.editor import VideoFileClip
import openai
import argparse
import tqdm 


parser = argparse.ArgumentParser(description="getting text fragments")
parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--xy", required=True,)
args = parser.parse_args()
openai.api_key = args.api_key

def annotation(full_text, text):
    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent medical professional specializing in concise communication and observation. You will be provided with: \
                            1) A structured explanation of an entire video to give you the larger context. \
                            2) Smaller segments of text from the video to summarize. \
                            Your task is to summarize each smaller segment of text into a single, clear, and concise sentence. \
                            Use the larger structured explanation as context to ensure the summary is accurate, relevant, and clinically meaningful.\
                            Always generate a summary based on the information provided, focusing on key actions, findings, or observations."
                    },
                    {
                        "role": "user",
                        "content":
                            f"Here is the text that will be used to extract key infomration from. \n \
                            structured explanation of an entire video:  {full_text} \n \
                            \n \
                            small segmented text: {text}"
                    }
                ]
            )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        return response_message

    except Exception as e:
        print(f"Error processing file {e}")


def transcribe_video_segment(video_filename, audio_filename):

    # Extract audio from the video segment
    video = VideoFileClip(video_filename)
    video.audio.write_audiofile(audio_filename,logger=None)
    
    # Transcribe audio
    result = whisper_model.transcribe(audio_filename)
    os.remove(audio_filename)  # Remove the temporary audio file
    return result['segments']

def get_bracket(keys):
    bracket= {
        "6":"0-6","12":"6-12",
        "18":"12-18","24":"18-24",
        "30":"24-30","36":"30-36",
        "42":"36-42","48":"42-48",
        "54":"48-54","60":"54-60"
    }
    final_brackets = {}
    for k,v in bracket.items():
        temp = 1000
        final_key = ""
        for key in keys:
            start, end = key.split(":")
            if round(float(end)) >= int(k):
                if float(end) < temp:
                    temp = float(end)
                    final_key = key

        final_brackets[v]=final_key

    return final_brackets

if __name__ == "__main__":

    audio_texts_path = "/data/shared/gauravs/llapsa/llapsa_encoded_video_clips/audio_texts"
    os.makedirs(audio_texts_path, exist_ok=True)

    # whisper model
    whisper_model = whisper.load_model("base")

    x, y = args.xy.split("-")

    video_clip_path = "/data/shared/gauravs/llapsa/vcgpt_clips"
    video_clips = [v for v in os.listdir(video_clip_path) if "_60sec_" in v][int(x):int(y)]

    # getting video_clips
    for vclip in tqdm.tqdm(video_clips, total=len(video_clips)):
    # vclip = "aneurysm_389_60sec_part14.mp4"
        output_file = f"{audio_texts_path}/{vclip.replace('.mp4', '.json')}"
        if not os.path.exists(output_file):
            try:
                audiofile_name = f"{audio_texts_path}/{vclip.replace('.mp4', '.wav')}"
                audio_data = transcribe_video_segment(f"{video_clip_path}/{vclip}", audiofile_name)

                full_text_path = f"/data/shared/gauravs/llapsa/additional_inbetween_data/structured_texts_from_gpt/{vclip.replace('.mp4', '.txt')}"
                full_text = open(full_text_path).readlines()[0]

                text_fragments = {}
                for ad in audio_data:
                    start, end = ad["start"], ad["end"]
                    _text = ad["text"]
                    if len(_text.strip().split()) > 2:
                        text_fragments[f"{start} : {end}"] = annotation(full_text, ad["text"])
                
                final_brackets = get_bracket(text_fragments.keys())

                # print(vclip)
                # print(final_brackets)
                # print(text_fragments)
                final_dict = {key: text_fragments[value] for key, value in final_brackets.items() if len(value.split()) > 2}

                with open(output_file, "w") as file:
                    json.dump(final_dict, file, indent=2)            

            except Exception as e:
                print(f"Error processing file {e}")