from transformers import pipeline
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-0.5B"

def main():
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
        
    pipe = pipeline("text-generation", model=MODEL_NAME_OR_PATH, tokenizer=tokenizer)
    # LOOP START
    while True:
        question = input("Please enter your question: ")
        prompt = f"Answer the following question. The last line of your response should be of the following format: 'Answer: $YOUR_ANSWER'. Think step by step before answering\n\nQuestion: {question}\nAnswer:"
        dataset = [prompt]
        response = pipe(dataset,  max_new_tokens=2048, batch_size=1)
        print("##########\n #RESPONSE\n##########\n\n")
        #print(prompt)
        print(response[0][0]['generated_text'])

if __name__ == "__main__":
    main()