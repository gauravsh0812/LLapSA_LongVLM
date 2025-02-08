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
                            "You are an expert evaluator assessing the **contextual accuracy** of AI-generated responses for surgical video-based question-answer pairs. \
                            Your task is to determine whether the predicted answer is contextually relevant, maintaining alignment with the video content and the correct answer.\n\n"
                            
                            "### **Evaluation Criteria:**\n"
                            "1 **Contextual Relevance**: The predicted answer should accurately reflect the surgical procedure shown in the video and not introduce unrelated or out-of-context information.\n"
                            "2 **Alignment with Main Themes**: The response should capture key themes and essential details relevant to the surgical process without misrepresenting or omitting critical aspects.\n"
                            "3 **Paraphrasing & Synonyms**: Accept variations in wording, as long as they **preserve the intended meaning and context** of the correct answer.\n"
                            "4 **No Hallucinations**: The response should not contain fabricated details or unrelated surgical concepts that do not appear in the video context.\n\n"

                            "### **Scoring System (1-5):**\n"
                            "**5 (Perfect)**: Fully aligned with the context, no out-of-place information.\n"
                            "**4 (Minor Misalignment)**: Mostly correct but contains **slight contextual deviations**.\n"
                            "**3 (Moderate Contextual Gaps)**: Some parts are off-topic **or missing relevant context**.\n"
                            "**2 (Major Contextual Errors)**: Largely misaligned with the video’s themes.\n"
                            "**1 (Completely Out of Context)**: Entirely irrelevant or misleading.\n\n"

                    },

                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
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

    #x, y = 4000,6000
    #n = 2

    pred_path=args.predicted_file_path
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/context", exist_ok=True)
    os.makedirs(f"{args.output_dir}/context/scores", exist_ok=True)

    mtype = "turbo3" if args.openai_model == "gpt-3.5-turbo" else "mini4o"
    #scores = open(f"{args.output_dir}/context/context_scores_{mtype}_{n}.lst", "w")
    len_scores = 0
    total_score = 0

    didnot_work = 0

    all_scr_files = os.listdir(f"{args.output_dir}/context/scores")
#    all_vids = []
#    for asf in all_scr_files:
#        all_vids.append(asf.split("--").strip()[0])

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        qtn = pc["question"]
        ans = pc["answer"]
        pred = pc["pred"]
        vid = pc["video_id"]
        if f"{vid}.txt" not in all_scr_files:
            try:
                response = annotate(qtn, pred, ans)
                scr = response['score']
                len_scores+=1
                total_score += scr
                with open(f"{args.output_dir}/context/scores/{vid}.txt", "w") as f:
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