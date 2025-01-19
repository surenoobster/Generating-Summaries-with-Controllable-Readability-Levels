from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scorer import FleschScorer
#from lookahead import Lookahead
from generationnew import Generator , Lookahead
from tqdm import tqdm
import argparse
import json

# Hardcoded single-line document and prompt
document = {
    "input_text": "British No1 Andy Murray was unsurprisingly in good form after his straight sets victory over John Isner on Sunday secured Great Britain's pathway into the Davis Cup quarter-finals. Although, that wasn't too be good news for his Great Britain team-mate Dominic Inglot - as Murray unintentionally stitched the 29-year-old up in a post-match interview. Speaking to Eurosport presenter Annabel Croft about how they would all celebrate the victory over the United States, Murray hinted that Inglot might spend the evening with a 'little girlfriend'. Andy Murray (third left) told Eurosport that Dominic Inglot (second right) had a 'little girlfriend' Murray (third left) cries with laughter as Inglot (second right) looks embarrassed to say the least . The doubles player (middle) then revealed he actually had a girlfriend at home who would be watching this . Murray cheekily said: 'Dom's got a little girlfriend on the go,' before going on to laugh. When Inglot was then asked immediately after who the girl was he replied somewhat embarrassingly: 'You've actually landed me in this. Because, I've actually got a girlfriend whose going to be watching this at home!' Murray then erupted with laughter, before going over to comfort a clearly embarrassed and shell-shocked Inglot. Murray can't contain himself as he erupts in laughter after finding out Inglot has a girlfriend . The British No 1 (third right) goes over to comfort the clearly embarrassed Inglot (second right) Earlier in the day Andy Murray secured a straight sets win over John Isner to send Great Britain into the Davis Cup quarter-finals - where they will face France . Sportsmail understands that the supporter infatuated by Inglot has been watching the matches over the weekend and her admiration for him was noted by the Great Britain team. Murray also wasn't aware that Inglot had a girlfriend. Inglot later took to Twitter, writing: 'Just to clarify, there is no girl on the side. The joke interview was just a joke. All banter with @andy_murray.' The doubles player - whose partner in Glasgow was Andy's older brother Jamie - took to social media on Sunday night to confirm it was a joke by Murray, and that he doesn't have a girl on the side."
}

parser = argparse.ArgumentParser()

# Base decoding model
parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
parser.add_argument("--cache_dir", type=str, default="./cache")
parser.add_argument("--output_file", type=str, required=True)

# Base decoding configuration
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--score", type=int, default=30)
parser.add_argument("--prompt", type=str, default="Write this for a FRE score 45")
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_input_length", type=int, default=1024)
parser.add_argument("--max_output_length", type=int, default=256)
parser.add_argument("--do_sample", action='store_true', default=True)

# Lookahead configuration
parser.add_argument("--do_lookahead", action="store_true", default=True)
parser.add_argument("--lookahead_length", type=int, default=64)
parser.add_argument("--lookahead_lambda", type=int, default=25)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--lookahead_decoding_type", type=str, default="greedy", choices=["greedy","beam","sample"])
parser.add_argument("--lookahead_beam", type=int, default=1)

# Scorer configuration
parser.add_argument("--scorer_model_type", type=str, default="roberta-large")
parser.add_argument("--scorer_num_layers", type=int, default=17)

args = parser.parse_args()

# Loading model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = model.cpu()  # Can optionally call .half() for mixed precision

# Preparing input
documents = [args.prompt + document["input_text"]]

scorer = FleschScorer('flesch', args.score)

# Create lookahead
lookahead = None
if args.do_lookahead:
    print("Creating Lookahead object with the following parameters:")
    print(f"lookahead_length={args.lookahead_length}, lookahead_lambda={args.lookahead_lambda}, lookahead_top_k={args.top_k}, decoding_type={args.lookahead_decoding_type}, num_beams={args.lookahead_beam}, num_return_sequences={args.lookahead_beam}, max_length={args.max_output_length}")
    
    lookahead = Lookahead(
        model,
        tokenizer,
        scorer,
        lookahead_length=args.lookahead_length,
        lookahead_lambda=args.lookahead_lambda,
        lookahead_top_k=args.top_k,
        decoding_type=args.lookahead_decoding_type,
        num_beams=args.lookahead_beam,
        num_return_sequences=args.lookahead_beam,
        max_length=args.max_output_length,
    )

# Create generator with lookahead
print("Creating Generator object...")
generator = Generator(model, lookahead=lookahead)

summaries = []

for i in tqdm(range(0, len(documents), args.batch_size)):
    print(f"Processing batch {i} to {i + args.batch_size}")
    input_str = documents[i:i + args.batch_size]

    # Tokenize inputs
    print(f"Tokenizing {len(input_str)} documents...")
    inputs = tokenizer(input_str, max_length=args.max_input_length, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to GPU
    inputs = {k: v.cpu() for k, v in inputs.items()}
    
    # Check inputs
    print(f"Tokenized inputs: {inputs['input_ids'].shape}")
    print(f"Tokenized input ids (first 5 tokens): {inputs['input_ids'][:5]}")

    # Generate output
    print("Generating output using the generator...")
    # Try generating directly using the model
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_output_length,
        do_sample=args.do_sample,
    )

    print(f"Direct model generated output: {output}")

        
    print(f"Generated output: {output}")
    
    '''if output is None:
        raise ValueError("The model did not generate any output. Check the input and model configuration.")'''
    
    # Decode output
    print("Decoding output...")
    output = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Handle output based on num_return_sequences
    if args.num_return_sequences == 1:
        summaries += output
    else:
        for j in range(0, len(output), args.num_return_sequences):
            summaries.append(output[j:j + args.num_return_sequences])

# Save file
print(f"Saving output to {args.output_file}...")
with open(args.output_file, "w") as f:
    if args.num_return_sequences == 1:
        for line in summaries:
            f.write(line + "\n")
    else:
        print("Dumping summaries as JSON...")
        json.dump(summaries, f)
