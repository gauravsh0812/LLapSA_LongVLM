messages=[
{
"role": "system",
"content":
"You are an expert evaluator in surgical procedures. Your task is to assess the factual accuracy, completeness, and medical soundness of AI-generated responses to surgical video-based questions.


        "**Evaluation Criteria:**\n"
        "1️⃣ **Factual Accuracy & Relevance**: The prediction must align with the correct answer regarding surgical principles, anatomy, and procedural details. **No factual errors or misinterpretations.**\n"
        "2️⃣ **Completeness**: All critical surgical details (e.g., anatomical landmarks, complications, surgical steps) should be included. Missing key details lowers the ranking.\n"
        "3️⃣ **Terminology & Synonyms**: Acceptable if medically equivalent. **Incorrect substitutions (e.g., 'cut' instead of 'coagulate') reduce accuracy.**\n"
        "4️⃣ **Clinical Logic**: The prediction should reflect correct surgical decision-making and avoid misleading statements.\n\n"
        
        "**Selection Categories:**\n"
        "✅ **Best**: The response is the most factually accurate, complete, and medically precise.\n"
        "⚠️ **Better**: The response is mostly correct but may lack minor surgical details.\n"
        "❌ **Worst**: The response contains major inaccuracies or misrepresents the procedure.\n\n"
        
        "**Task:**\n"
        "Compare multiple predicted answers to the correct answer using these guidelines. Categorize each prediction as **Best, Better, or Worst** based on its quality. Output the result in **strict JSON format**.\n\n"
        "**Response Format:**\n"
        "{ 'best': <predX>, 'better': <predY>, 'worst': <predZ> }\n\n"
        "**DO NOT provide explanations or extra text.**"
},

{
    "role": "user",
    "content": 
        "Evaluate the following surgical video-based question-answer pair for factual accuracy:\n\n"
        f"Question: {qtn}\n"
        f"Correct Answer: {ans}\n"
        f"Predicted Answers: {pred1}, {pred2}, {pred3}\n\n"
        "Return **only** a Python dictionary in the format:\n"
        "{ 'best': <predX>, 'better': <predY>, 'worst': <predZ> }\n\n"
        "**Example valid response:** { 'best': pred1, 'better': pred2, 'worst': pred3 }\n"
        "**Do not include explanations, additional text, or formatting.**"
}

]



CONTEXT

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

                            "### **Task:**\n"
                            "Compare the predicted answer with the correct answer based on the criteria above. Determine if the prediction maintains contextual accuracy and remains within the scope of the video.\n\n"
                            
                            "**Response Format:**\n"
                            "{'contextual_alignment': '<Aligned / Partially Aligned / Misaligned>'}\n\n"
                            
                            "**Strictly return only the JSON dictionary format.**\n"
                            "**Example valid response:** {'contextual_alignment': 'Aligned'}"
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