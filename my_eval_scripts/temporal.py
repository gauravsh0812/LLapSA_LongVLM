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
                            "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                            "- Evaluate the temporal accuracy of the prediction compared to the answer."
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