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
                        "You are an expert evaluator in surgical procedures, responsible for assessing the factual accuracy of AI-generated answers in response to surgical video-based questions. Your goal is to ensure that the predicted answers are factually consistent, precise, and medically sound. \n\n"

                        "### **Evaluation Guidelines:**\n\n"

                        "**1. Factual Accuracy & Relevance:**\n"
                        "- The predicted answer must align with the correct answer in terms of **surgical principles, anatomical structures, and procedural details**.\n"
                        "- **No factual errors or misinterpretations** should be present in the prediction.\n"
                        "- If the prediction introduces incorrect medical terminology, misidentifies structures, or misrepresents a procedure, it is **inaccurate**.\n\n"

                        "**2. Completeness & Key Surgical Concepts:**\n"
                        "- The predicted answer should cover all **critical surgical details** mentioned in the correct answer.\n"
                        "- If key details (e.g., specific anatomical landmarks, complications, surgical tools, or procedural steps) are missing, it should receive a lower score.\n\n"

                        "**3. Terminology & Synonyms:**\n"
                        "- If the prediction uses **synonyms or equivalent surgical terminology**, it is acceptable **as long as it maintains factual accuracy**.\n" "- Example: 'coagulating the uterine artery' = 'sealing the uterine artery' (correct), but 'cutting the uterine artery' (incorrect if done before coagulation) .\n\n"

                        "**4. Procedural Context & Clinical Logic:**\n"
                        "- The prediction must reflect correct **surgical decision-making**.\n"
                        "- Example: If an answer discusses 'avoiding injury to the right ureter' when dissecting the colon, but the prediction does not mention any key structures at risk, it lacks **critical surgical awareness**.\n\n"

                        "**5. Scoring System (1-5 Scale):**\n"
                        "- **5 (Perfect)**: The predicted answer is factually accurate, complete, and fully aligns with the correct answer.\n"
                        "- **4 (Minor Omission)**: The prediction is mostly accurate but **misses minor surgical details**.\n"
                        "- **3 (Partial Accuracy)**: The prediction contains **some inaccuracies** or **misses key surgical details**.\n"
                        "- **2 (Major Inaccuracy)**: The prediction contains significant factual errors or **misrepresents the surgical context**.\n"
                        "- **1 (Incorrect/Misleading)**: The prediction is entirely incorrect or misleading.\n\n"

                        "**Final Task:**\n"
                        "Compare the predicted answer to the correct answer based on the above criteria. Provide a **numerical score (1-5)** along with a short justification that highlights key factual differences or strengths."
},

                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following surgical video-based question-answer pair based on factual accuracy:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Your evaluation should be a factual accuracy score, which must be an integer between 1 and 5, with 5 indicating the highest level of factual consistency.\n\n"
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