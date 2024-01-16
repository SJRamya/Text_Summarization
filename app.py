from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from rouge_score import rouge_scorer

app = Flask(__name__)

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        input_text = request.form.get('inputtext_')
        print("Received input text:", input_text)
        received_summary = request.form.get('summarytext_')
        tokenized_text = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        print(1)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        print(2)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        print(3)
        # Calculate ROUGE scores 
        rouge_scores = scorer.score(received_summary, summary)
        print(4)
        print('\nrouge1:',rouge_scores['rouge1'].fmeasure , '\nrouge2:',rouge_scores['rouge2'].fmeasure , '\nrougel:',rouge_scores['rougeL'].fmeasure)
        return render_template('output.html', result=summary, rouge1=round(rouge_scores['rouge1'].fmeasure*100,2) , rouge2=round(rouge_scores['rouge2'].fmeasure*100,2) , rougel=round(rouge_scores['rougeL'].fmeasure*100,2))

if __name__ == '__main__':
    app.run()