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
    try:
        # Compute the temporal understanding score
            completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an expert evaluator assessing the **temporal understanding** of AI-generated responses for surgical video-based question-answer pairs. Your task is to determine whether the predicted answer correctly maintains the **sequence of events** as they occur in the surgical procedure.\n\n"

                            "### **Evaluation Criteria:**\n"
                            "1 **Correct Order of Events**: The predicted answer must follow the correct chronological sequence of surgical steps, complications, or anatomical progressions.\n"
                            "2 **Temporal Consistency**: The response should avoid mixing up steps, skipping ahead, or presenting details out of order.\n"
                            "3 **Paraphrasing with Temporal Accuracy**: Synonyms and paraphrases are acceptable **only if they preserve the correct event sequence**.\n"
                            "4 **No Misalignment or Reordering**: If the predicted answer introduces incorrect sequencing (e.g., describing suturing before an incision), it should be rated lower.\n\n"

                            "### **Scoring System (1-5):**\n"
                            "**5 (Perfect)**: Fully accurate sequence, no temporal errors.\n"
                            "**4 (Minor Sequencing Issue)**: Mostly correct but **minor steps slightly out of order**.\n"
                            "**3 (Moderate Temporal Errors)**: Some sequence mistakes **or missing steps**.\n"
                            "**2 (Major Sequencing Errors)**: Many steps out of order **or incorrect transitions**.\n"
                            "**1 (Completely Disordered)**: Entirely incorrect sequence, disrupting understanding.\n\n"
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
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

    pred_path="inference_longvlm.json"
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/temporal", exist_ok=True)
    os.makedirs(f"{args.output_dir}/temporal/scores", exist_ok=True)

    mtype = "turbo3" if args.openai_model == "gpt-3.5-turbo" else "mini4o"

    #scores = open(f"outputs/correctness/eval_1_correctness_scores_turbo3.lst", "w")
    len_scores = 0
    total_score = 0

    didnot_work = 0

    all_scr_files = os.listdir("outputs/temporal/scores")

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

                with open(f"outputs/temporal/scores/{vid}.txt", "w") as f:
                    f.write(f"{vid} -- {scr}")

                #scores.write(f"{vid} \t score: {scr} \n")

            except:
                print(f"{vid}.txt not working!")
                didnot_work+=1

    average_score = total_score / len_scores

    print("Average score for correctness:", average_score)
    print("didnot work: ", didnot_work)

if __name__ == "__main__":
    main()