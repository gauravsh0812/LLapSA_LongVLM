import openai, os
import argparse
import tqdm
import json
import ast

parser = argparse.ArgumentParser(description="Training")

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
                            "You are an expert evaluator assessing the **detail orientation** of AI-generated responses for surgical video-based question-answer pairs. \
                            Your task is to determine whether the predicted answer provides a **complete and specific** response aligned with the surgical context.\n\n"
                            
                            "### **Evaluation Criteria:**\n"
                            "1 **Coverage of Key Surgical Details**: The predicted answer should include all major procedural steps, anatomical structures, and relevant medical considerations from the correct answer.\n"
                            "2 **Specificity vs. Generalization**: The response should provide **precise and relevant** details rather than vague or overly general statements.\n"
                            "3 **Terminology & Synonyms**: Accept different phrasing **only if it retains the same level of specificity and accuracy** as the correct answer.\n"
                            "4 **No Missing Critical Information**: Any omission of essential surgical details (e.g., key instruments, anatomical landmarks, or safety precautions) should lower the evaluation.\n\n"

                            "### **Scoring System (1-5):**\n"
                            "**5 (Perfect)**: Fully detailed and specific, no missing key elements.\n"
                            "**4 (Minor Omission)**: Mostly complete, but **lacks minor details**.\n"
                            "**3 (Moderate Gaps)**: Some key details are missing **or too vague**.\n"
                            "**2 (Major Gaps in Detail)**: Largely incomplete or too generalized.\n"
                            "**1 (Lacking Detail Entirely)**: Barely provides any relevant details.\n\n"
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        #print(response_message)
        response_dict = ast.literal_eval(response_message)
        return response_dict

    except Exception as e:
        print(f"Error processing file {e}")

def main():

    """
    Main function to control the flow of the program.
    """

    #x, y = 4000,6000
    #n = 2

    pred_path=args.predicted_file_path
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/detailed_orientation", exist_ok=True)
    os.makedirs(f"{args.output_dir}/detailed_orientation/scores", exist_ok=True)
    mtype = "turbo3" if args.openai_model == "gpt-3.5-turbo" else "mini4o"
#    scores = open(f"{args.output_dir}/detailed_orientation/detailed_orientation_scores_{mtype}_{n}.lst", "w")
    len_scores = 0
    total_score = 0

    didnot_work = 0

    all_scr_files = os.listdir(f"{args.output_dir}/detailed_orientation/scores")

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):

        qtn = pc["question"]
        ans = pc["answer"]
        pred = pc["pred"]
        vid = pc["video_id"]
        if f"{vid}.txt" not in all_scr_files:
            try:
                #print(qtn)
                #print(ans)
                #print(pred)
                response = annotate(qtn, pred, ans)
                #print(response)
                scr = response['score']
                len_scores+=1
                total_score += scr

                with open(f"{args.output_dir}/detailed_orientation/scores/{vid}.txt", "w") as f:
                    f.write(f"{vid} -- {scr}")

            #scores.write(f"{vid} \t score: {scr} \n")

            except:
                didnot_work+=1
                print(f"{vid}.txt not working!")

    average_score = total_score / len_scores

    print("Average score for correctness:", average_score)
    print("didnot work: ", didnot_work)

if __name__ == "__main__":
    main()
