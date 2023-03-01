import os
import pandas as pd
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
file = pd.read_csv('extracted_sentences.csv')
print(file.columns)
sentences = file['1']
pt = file['0']

from transformers import BertTokenizer

model = torch.load('entire_model.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# Define a sample sentence
sentence = "This is an example sentence."

# Tokenize the sentence
tokens = tokenizer.encode_plus(sentence, add_special_tokens=True,
                               return_tensors='pt', padding=True,
                               truncation=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokens = {key: val.to(device) for key, val in tokens.items()}

# Move the model to the same device as the input tensor
model.to(device)

# Make a prediction on the tokenized sentence
predictions = []
sentencs = []
paths = []
with torch.no_grad():
    for sen, p in zip(sentences, pt):
        tokens = tokenizer.encode_plus(sen, add_special_tokens=True,
                                       return_tensors='pt', padding=True,
                                       truncation=True)
        tokens = {key: val.to(device) for key, val in tokens.items()}
        outputs = model(**tokens)
        print(outputs)
        predicted_label = torch.argmax(outputs[0]).item()
        predictions.append(predicted_label)
        sentencs.append(sen)
        paths.append(p)

        print(predicted_label)

        predictions_df = pd.DataFrame({'path': paths,
                                       'sentence': sentencs,
                                       'prediction': predictions})
        predictions_df.to_csv('predictions.csv')
# model.eval()
# print(f"Predicted label for the sentence '{sentence}': {predicted_label}")
# predictions_df=pd.DataFrame(predictions)
# predictions_df.to_csv('predictions.csv')
