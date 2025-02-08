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
                        "You are an expert evaluator in surgical procedures. Your task is to assess the factual accuracy, completeness, and medical soundness of AI-generated responses to surgical video-based questions. \n\n"
            
                        "**Evaluation Criteria:**\n"
                        "1 **Factual Accuracy & Relevance**: The prediction must align with the correct answer regarding surgical principles, anatomy, and procedural details. **No factual errors or misinterpretations.**\n"
                        "2 **Completeness**: All critical surgical details (e.g., anatomical landmarks, complications, surgical steps) should be included. Missing key details lowers the score.\n"
                        "3 **Terminology & Synonyms**: The annswer is acceptable if medically equivalent. Any incorrect substitutions (e.g., 'cut' instead of 'coagulate') will reduce accuracy.\n"
                        "4 **Clinical Logic**: The prediction should reflect correct surgical decision-making and avoid misleading statements.\n\n"
                        
                        "**Scoring System (1-5):**\n"
                        "**5 (Perfect)**: Fully accurate, complete, and medically precise.\n"
                        "**4 (Minor Omission)**: Mostly correct but **lacks minor surgical details**.\n"
                        "**3 (Partial Accuracy)**: Some inaccuracies **or missing key details**.\n"
                        "**2 (Major Inaccuracy)**: Contains significant errors **or misrepresents the procedure**.\n"
                        "**1 (Incorrect/Misleading)**: Entirely incorrect or misleading.\n\n"
            
                    },

                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following surgical video-based question-answer pair based on factual accuracy:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Your evaluation should be a factual accuracy score, which must be an integer between 1 and 5, based on the instructions provided.\n\n"
                            "Return your response **only** as a Python dictionary string in the following format:\n"
                            "{'score': <integer>}\n\n"
                            "DO NOT provide any additional text, explanations, or formatting. Only output the Python dictionary string.\n\n"
                            "Example of a valid response: {'score': 4}"
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
        try:
            qtn = pc["question"]
            ans = pc["answer"]
            pred = pc["pred"]
            vid = pc["video_id"]
            if f"{vid}.txt" not in all_scr_files:
                response = annotate(qtn, pred, ans)
                scr = response['score']
                len_scores+=1
                total_score += scr

                with open(f"{args.output_dir}/correctness/scores/{vid}.txt", "w") as f:
                    f.write(f"{vid} -- {scr}")
        
        except:
            didnot_work+=1
            print(f"{vid}.txt not working!")

    average_score = total_score / len_scores

    print("Average score for correctness:", average_score)
    print("didnot work: ", didnot_work)

if __name__ == "__main__":
    main()