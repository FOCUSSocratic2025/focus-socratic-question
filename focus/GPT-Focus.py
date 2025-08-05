import pandas as pd
import re
from google.colab import files
import openai
import os

# (Ensure your OPENAI_API_KEY is set in the environment; e.g., in Colab: 
#    import os; os.environ["OPENAI_API_KEY"] = "your_key_here"
# )

# Upload prompt file
print("📥 Please upload 'Socratic_Questioning_Integrated_Prompt.txt'")
uploaded_prompt = files.upload()
prompt_file = list(uploaded_prompt.keys())[0]

# Load the prompt
with open(prompt_file, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Upload CSV file
print("📥 Now upload your CSV file (must contain arguments)")
uploaded_csv = files.upload()
csv_file = list(uploaded_csv.keys())[0]

# Load CSV and check column
df = pd.read_csv(csv_file)
print("📄 Columns in your CSV:", list(df.columns))

if "Argument" not in df.columns:
    print("⚠️ 'Argument' column not found.")
    arg_column = input("👉 Type the name of the column that contains arguments: ").strip()
    if arg_column not in df.columns:
        raise ValueError(f"❌ Column '{arg_column}' not found in CSV.")
else:
    arg_column = "Argument"

# Chat completion function (uses openai package and environment variable)
def classify_argument_fsq(argument_text, model="gpt-4"):
    user_prompt = f"Argument:\n{argument_text}"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return "Classification:\n[]\n\nSpan:\n[]"

# Extract FSQ types and spans
def extract_predictions(gpt_output):
    try:
        classification = re.search(r"Classification:\s*\[(.*?)\]", gpt_output, re.DOTALL)
        spans = re.search(r"Span:\s*\[(.*?)\]", gpt_output, re.DOTALL)

        types = [x.strip().strip('"') for x in classification.group(1).split(",")] if classification else []
        span_texts = [x.strip().strip('"') for x in spans.group(1).split(",")] if spans else []

        types += [None] * (2 - len(types))
        span_texts += [None] * (2 - len(span_texts))

        return types[0], types[1], span_texts[0], span_texts[1]
    except:
        return None, None, None, None

# Run classification on all rows
predicted_1, predicted_2, span_1, span_2 = [], [], [], []

for arg in df[arg_column]:
    gpt_output = classify_argument_fsq(arg)
    t1, t2, s1, s2 = extract_predictions(gpt_output)
    predicted_1.append(t1)
    predicted_2.append(t2)
    span_1.append(s1)
    span_2.append(s2)

# Save result
df["Predicted_Type1"] = predicted_1
df["Predicted_Type2"] = predicted_2
df["Span_PredictedType1"] = span_1
df["Span_PredictedType2"] = span_2

output_path = "/content/new-SQ-testset-GPT4-1shot.csv"
df.to_csv(output_path, index=False)
print(f"✅ Done! Results saved to: {output_path}")
files.download(output_path)