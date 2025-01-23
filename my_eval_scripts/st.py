import openai, os, sys
import argparse
import time
import json
import ast
import multiprocessing as mp

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--input_file", required=True, help="Input json file containing all the obseravtion,Note,\
                                                            plan, reasons, equipemnets, and organ details.")
parser.add_argument("--output_json_file_dir", required=True, help="output file directory path")
parser.add_argument("--xy", required=True)
args = parser.parse_args()

openai.api_key = args.api_key

def msg_system(key):
    messages_system= {
        "observation": "You are a highly specialized AI assistant focused on surgical education. \
                        You will be provided with a detailed text description or transcription of a surgical video clip from a lecture, along with specific 'observations' noted by a medical professional. \
                        You do not have access to the actual video. Your task is to generate a high-quality question-and-answer (Q&A) pair that reflect insights based on these observations. \
                        Each question should align with the medical professional's perspective, aiming to explore critical aspects, techniques, or anatomical details as they might be understood or highlighted in the actual procedure. \
                        Avoid referencing text details like the title or description directly. Structure your responses as if both you and the User are jointly observing the procedure to create a dynamic, informative dialogue.",

        "reason":   "You are an expert AI assistant specializing in surgical education. \
                    Provided with a text description or transcription of a surgical video clip from a lecture, as well as the 'reasons' behind key actions noted in the description, you will create a question-and-answer pair based on these insights. \
                    Although you cannot view the actual video, generate questions that delve into the reasoning and purpose behind each surgical step, aiming to uncover the decision-making process, techniques, and anatomical considerations. \
                    The question should reflect a unique aspect of the reasoning process and offer a fresh perspective, as if you and the User are observing the surgery together in real time. \
                    Avoid referring to textual elements like the title or description directly. Instead, explore each question from a different angle to provide comprehensive, varied insights into the surgical approach.",

        "plan":     "You are an AI assistant with expertise in surgical education. \
                    You will be provided with a detailed text description or transcription of a surgical video clip from a lecture, along with a 'plan' outlining upcoming steps in the procedure. \
                    Although you do not have access to the actual video, your role is to create a question-and-answer pair that explore the rationale and considerations behind planning each next step. \
                    The questions should encourage thinking about how and why specific actions are anticipated or sequenced. \
                    The question should be framed as if you and the User are observing and discussing the surgical plan in real time, with a focus on future actions, expected outcomes, and procedural strategy. \
                    Avoid direct references to textual elements like the title or description. Instead, emphasize a forward-looking perspective to generate unique, insightful questions about the surgical approach and its objectives.",

        "note":     "You are an AI assistant with a focus on surgical education. \
                    You are given a text description or transcription of a surgical video clip from a lecture, along with an important 'note' explaining a particular action taken by the medical professional. \
                    While you do not have access to the actual video, your task is to generate a question-and-answer pair that highlight the significance of these notes in understanding the surgical steps, techniques, or safety considerations. \
                    The question should emphasize the key points or expert tips as though you and the User are actively observing the surgery. \
                    Structure questions to explore why specific details or nuances in the note are essential for the surgical outcome. \
                    Avoid direct references to elements like titles or descriptions; instead, focus on the expert insights that the note brings to light, adding value to each step in the procedure.",

        "description": "You are an AI assistant specialized in surgical topics. Using the provided details, create a description of the surgical procedure. \
                        Include the sequence of actions, observations made during each step, the overall plan guiding the surgery, and the reasons behind each maneuver. \
                        Ensure that the description highlights the surgeon's approach to carefully cleaning, dissecting, and identifying critical anatomical structures while maintaining precision and minimizing disturbance. \n \
                        The details that will be provided by the User are: \
                            Transcript: The text description of the surgical video. \
                            Observations: The observations made form  the text description. \
                            Plan: The plans for the next step or future actions mentioned by surgeon in the video. \
                            Reason: The reasons provided by the surgeon for the actions/steps taken in the video during the surgery. \
                            Note: Any important point mentioned by the surgeon which should be taken care of while performing the surgery. \n \
                        Using these elements, create a question-answer pair to provide a cohesive and informative description of the surgery."

    }
    return messages_system[key]

def msg_user(key, text, detail):

    if key == "observation":
        prompt = f"Provide me the json dictionary of a Q&A pair,  for the given text and obseravtion associated with it. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the observation made by the medical professional: {detail}",

    elif key == "reason":
        prompt = f"Provide me the json dictionary of a Q&A pair,  for the given text and the reason provided. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the reason provided by the medical professional for the action: {detail}",

    elif key == "plan":
        prompt = f"Provide me the json dictionary of  a Q&A pair,  for the given text and plan for the next steps. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the plan provided by the medical professional for the future actions: {detail}",

    elif key == "note":
        prompt = f"Provide me the json dictionary of  a Q&A pair,  for the given text and important point mentioned by surgeon. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the important note mentioned by the medical professional: {detail}",

    elif key == "description":
        prompt = f"Based on the information provided down below, give the comprehensove description of the surgery. \
                        provide  the final reponse in the form of a question-answer pair. \
                        Here is the text description of the surgical video scenario: {text} \n\n \
                        Along with the text description, here are the list of the observations, reasons, plan, and notes related to the surgical video, mentioned by the surgeon. \
                        Observations: {detail['observations']} \n \
                        Reasons: {detail['reasons']} \n \
                        Plans: {detail['plans']} \n \
                        Notes: {detail['notes']}"

    return prompt

def annotate(key, text, detail):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """

    # try:
    # Compute the correctness score
    system_text = msg_system(key)
    user_text = msg_user(key, text, detail)
    completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role":"system",
                        "content":
                            f"""
                            {system_text}

                            #-------------#
                            #INSTRUCTIONS#

                            Below are requirements for generating the questions and answers in the conversation:
                            - Avoid directly quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as
                            these may reveal the conversation is based on the text information, rather than the video clip itself. Focus on
                            the visual aspects of the video that can be inferred without the text information.
                            - Do not use phrases like "mentioned", "title", "description" in the conversation. Instead, refer to the
                            information as being "in the video."

                            Your reply should be in the following json dictionary format where the key will be the question and the value will be the answer.
                            """
                    },
                    {
                        "role": "user",
                        "content":
                                f"{user_text}"
                    }
                ]
            )
    # Convert response to a Python dictionary.
    response_dict = completion["choices"][0]["message"]["content"]
    return response_dict

# except Exception as e:
#     print(f"Error processing file {e}")
#     return {}

def main_parallel(arr):

    (ind, af) = arr

    def get_response(key, text, o, video_id):
        response = annotate(key, text, o)
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "")
        response = ast.literal_eval(response)
        final_response = {}
        final_response["q"] = list(response.keys())[0]
        final_response["a"] = list(response.values())[0]
        final_response["video_id"] = video_id
        final_response["type"] = key

        return final_response

    def list_to_str(items):
        if len(items) > 1:
            output = ", ".join(items[:-1]) + ", and " + items[-1]
        else:
            output = items[0] if items else ""
        return output

    try:
        video_id = af["video_id"].split(".")[0]
        text = af["transcript"]

        # print("text: ", text + "\n")

        obs = af["observation"]
        rsn = af["reason"]
        pln = af["plan"]
        nt = af["note"]
        ogn = af["organs"]
        eqp = af["equipments"]

        # getting QA
        for i, o in enumerate(obs):
            response = get_response("observation", text,o, video_id)
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_obs{i}.json", "w") as f:
                f.write(json.dumps(response))
                f.close()

        for i,r in enumerate(rsn):
            response = get_response("reason", text,r, video_id)
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_rsn{i}.json", "w") as f:
                f.write(json.dumps(response))
                f.close()

        for i,p in enumerate(pln):
            response = get_response("plan", text,p, video_id)
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_pln{i}.json", "w") as f:
                f.write(json.dumps(response))
                f.close()
            # final description QA
            details = {}
            details['observations'] = obs
            details['reasons'] = rsn
            details['plans'] = pln
            details["notes"] = nt
            response = get_response("description", text, details, video_id)
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_detail.json", "w") as f:
                f.write(json.dumps(response))
                f.close()

        # adding quantative questions
        equipments = list_to_str(eqp)
        organs = list_to_str(ogn)

        if len(eqp) >=1:
            d = {
                "q": "What equipments are used in the surgical video?",
                "a":  f"The equipments used in the surgery are {equipments}.",
                "video_id": video_id,
                "type": "quantative",
                }
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_eqp.json", "w") as f:
                f.write(json.dumps(response))
                f.close()

        if len(ogn)>=1:
            d = {
                "q": "What organs are involved in the surgery?",
                "a":  f"The organs involved in the surgery are {organs}.",
                "video_id": video_id,
                "type": "quantative",
                }
            with open(f"/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/{ind}_org.json", "w") as f:
                f.write(json.dumps(response))
                f.close()


    except Exception as e:
        error_message = str(e)
        print(f"Error processing file {e}")
        if "Rate limit reached" in error_message or "You exceeded your current quota" in error_message:
            print("Rate limit reached. Stopping execution.")
            sys.exit(1)  # Or re-raise the exception if preferred

def clean_json_files(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all items in the directory
        for filename in os.listdir(directory_path):

            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)  # Directly remove the JSON file
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The directory {directory_path} does not exist.")


# def sample_generator(data):
#     for i in range(0, len(data), 20):
#         yield data[i:i+20]

def main():
    """
    Main function to control the flow of the program.
    """
    start_time = time.time()

    xy = args.xy
    x,y = xy.split("-")

    output_json_file_dir = args.output_json_file_dir

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    filtered_data = [item for item in data if "_45sec_" in item['video_id']]

    sorted_data = sorted(filtered_data, key=lambda x: x['video_id'])

    print("total data: ", len(sorted_data))

    if int(y)<=len(sorted_data):
        data = sorted_data[int(x):int(y)]
    else:
        data = sorted_data[int(x):]

    print(f"will run samples from {x} to {y}. Total samples to run: {int(y)-int(x)}")

    total_samples = len(data)
    tq,ogq,rq,nq,pq,eq,oq = 0,0,0,0,0,0,0

    for i,af in enumerate(data):
        try:
            obs = af["observation"]
            rsn = af["reason"]
            pln = af["plan"]
            nt = af["note"]
            ogn = af["organs"]
            eqp = af["equipments"]
            text = af["transcript"]
            if len(obs) > 0:
                for _ in obs: oq+=1
            if len(rsn) > 0:
                for _ in rsn: rq+=1
            if len(pln) > 0:
                for _ in pln: pq+=1
            if len(nt) > 0:
                for _ in nt: nq+=1
            if len(ogn) > 0:
                ogq+=1
            if len(eqp) > 0:
                eq+=1
            if len(text) > 0:
                tq+=1
        except:
            continue

    print("Total expected QA: ", tq+ogq+rq+nq+pq+eq+oq)

    # for batch in sample_generator(data):
    N = 100
    samples_done = 0
    smpl = 0

    while samples_done < total_samples:
        temp_arr = []
        for i,af in enumerate(data[smpl:smpl+N]):
            # for i, af in enumerate(batch):
            temp_arr.append((i, af))

        with mp.Pool(8) as pool:
            pool.map(main_parallel, temp_arr)

        all_responses = []
        # Read and combine all JSON files in the "temps" directory
        temp_dir = "/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/"
        for filename in os.listdir(temp_dir):
            if filename.endswith(".json"):
                with open(os.path.join(temp_dir, filename), 'r') as f:
                    try:
                        response = json.load(f)
                        all_responses.append(response)
                    except Exception as e:
                        print(f"Error processing file {e}")

        # Write all responses to the JSON file
        output_file = open(f"{output_json_file_dir}/{int(x)+smpl}_{int(x)+smpl+N}.json", "w")
        output_file.write(json.dumps(all_responses, indent=2))
        output_file.write('\n')  # Add a newline after the entire JSON object

        clean_json_files("/data/shared/gauravs/llapsa/csv_json_data/surgical_tutor/temps/")
        samples_done += N
        smpl += N
        temp_arr = []

        print(f"Total samples done: {samples_done} / {total_samples}")


    end_time = time.time()
    total_running_time = (end_time - start_time)/60  # Calculate the total running time
    print(f"Total running time: {total_running_time:.2f} minutes")  # Print the total running time

if __name__ == "__main__":
    main()


