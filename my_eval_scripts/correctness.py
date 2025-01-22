import openai
import os
import argparse
import tqdm
import json
import ast

parser = argparse.ArgumentParser(description="evaluation")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--openai_model", required=True, help="which openai model -- gpt-3.5-turbo or gpt-4o-mini")
parser.add_argument("--predicted_file_path", required=True, help="file containing the predictions from trained model (Inference output file)")
parser.add_argument("--output_dir", required=True, help="output directory")
args = parser.parse_args()

openai.api_key = args.api_key

def eliminate_repetitive(text):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """
    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                    "role": "system",
                    "content": "You are an intelligent chatbot tasked with removing repetitiveness from a given text. \
                                Follow these instructions:\n\n \
                                ## INSTRUCTIONS:\n \
                                - Preserve the main text as it is. \n \
                                - Eliminate similar or redundant sentences to simplify the final text.\n \
                                - DO NOT rewrite sentences; only remove extraneous ones."
                    },
                    {
                    "role": "user",
                    "content": 
                            f"Here is the text: {text}"
                    }

                ]
            )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        return response_message

    except Exception as e:
        print(f"Error processing file {e}")

def annotate(qtn, pred, ans):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """
    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content":
                                "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "

                                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"                                   "- The predicted answer must be factually accurate and align with the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the factual accuracy of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
        return response_dict

    except Exception as e:
        print(f"Error processing file {e}")

def main():

    """
    Main function to control the flow of the program.
    """

    pred_path=args.predicted_file_path
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/correctness", exist_ok=True)
    os.makedirs(f"{args.output_dir}/correctness/scores", exist_ok=True)

    len_scores = 0
    total_score = 0

    didnot_work = 0

    all_scr_files = os.listdir(f"{args.output_dir}/correctness/scores")

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        # try:
        print(len(pc))
        qtn = pc["question"]
        ans = pc["answer"]
        pred = pc["pred"]
        vid = pc["video_id"]
        # if f"{vid}.txt" not in all_scr_files:
        pred = eliminate_repetitive(pred)
        print(pred)
        exit()
        response = annotate(qtn, pred, ans)
        scr = response['score']
        len_scores+=1
        total_score += scr
        with open(f"{args.output_dir}/correctness/scores/{vid}.txt", "w") as f:
            f.write(f"{vid} -- {scr}")
        
        # except:
        #     didnot_work+=1
        #     print(f"{vid}.txt not working!")

    average_score = total_score / len_scores

    print("Average score for correctness:", average_score)
    print("didnot work: ", didnot_work)

if __name__ == "__main__":
    main()