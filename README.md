# Generating-Summaries-with-Controllable-Readability-Levels

```
import os
import json
from datasets import load_dataset

# Download the CNNDM data
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Create a directory to save the dataset locally
os.makedirs("local_dataset", exist_ok=True)

def save_dataset_locally(data, split):
    """
    Save the dataset locally as a JSON file.
    """
    file_path = f"local_dataset/{split}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Dataset saved locally to {file_path}")

def save_raw_dataset(split):
    """
    Extract and save the raw dataset (articles, summaries, and IDs) for the given split.
    """
    local_data = []
    for dial, sum, id_ in zip(dataset[split]['article'], dataset[split]['highlights'], dataset[split]['id']):
        local_data.append({"article": dial, "summary": sum, "id": id_})

    save_dataset_locally(local_data, split)

# Save each dataset split locally
save_raw_dataset('train')
save_raw_dataset('validation')
save_raw_dataset('test')
```

then for FRE and other criteria for minimum of 100 lines 

```
import json
import os
from datasets import load_dataset

# Update the paths to your local dataset files
data_files = {
    "train": "/home/surenoobster/Documents/controllable-readability-summarization/src/Local_Dataset/train.json",
    "validation": "/home/surenoobster/Documents/controllable-readability-summarization/src/Local_Dataset/validation.json",
    "test": "/home/surenoobster/Documents/controllable-readability-summarization/src/Local_Dataset/test.json"
}

# Load dataset from local files
dataset = load_dataset("json", data_files=data_files)

# Function to filter based on article and summary word counts
def filter_article_and_summary(examples, min_article_words=110, min_summary_words=110):
    """
    Filters examples to retain those where:
    - The article has at least `min_article_words`.
    - The summary has at least `min_summary_words`.
    """
    return [
        example
        for example in examples
        if len(example['article'].split()) >= min_article_words
        and len(example['summary'].split()) >= min_summary_words
    ]

# Apply filtering to each split (train, validation, and test)
filtered_train = dataset['train'].filter(
    lambda x: len(x['article'].split()) >= 110 and len(x['summary'].split()) >= 110
)
filtered_validation = dataset['validation'].filter(
    lambda x: len(x['article'].split()) >= 110 and len(x['summary'].split()) >= 110
)
filtered_test = dataset['test'].filter(
    lambda x: len(x['article'].split()) >= 110 and len(x['summary'].split()) >= 110
)

# Create a new directory for storing the filtered datasets
output_dir = "/home/surenoobster/Documents/controllable-readability-summarization/src/18Dec"
os.makedirs(output_dir, exist_ok=True)

# Define the output file paths
train_file = os.path.join(output_dir, "train_greater_110_article_10_summary.json")
validation_file = os.path.join(output_dir, "validation_greater_110_article_10_summary.json")
test_file = os.path.join(output_dir, "test_greater_110_article_10_summary.json")

# Save the filtered datasets to JSON files
filtered_train.to_json(train_file)
filtered_validation.to_json(validation_file)
filtered_test.to_json(test_file)

# Print confirmation of saved files
print(f"Filtered datasets saved to:")
print(f"Train: {train_file}")
print(f"Validation: {validation_file}")
print(f"Test: {test_file}")

# Optionally, inspect the first entry of the 'test' dataset after filtering
print("First entry from filtered test dataset:")
print(json.dumps(filtered_test[0], indent=4))

```



Download the dataset using this , then preprocess cnndm using

https://github.com/surenoobster/Generating_new_summaries/blob/main/preprocess_cnndm_2.py

then 

for category instruct finetuning preprocessing 

https://github.com/surenoobster/Generating_new_summaries/blob/main/generate_category_2.py

then for score instruct finetuning preprocessing

https://github.com/surenoobster/Generating_new_summaries/blob/main/generate_score_2.py


then after that 

```
import json
import os

# Define input and output file paths
input_file = ""  # Replace with the path to your JSON file
output_file = ""
# Process each line as a separate JSON object
with open(input_file, "r", encoding="utf-8") as infile:
    extracted_data = []
    for line in infile:
        try:
            data = json.loads(line.strip())
            extracted_data.append({
                "prompt": data["prompt"],
                "input_text": data["input_noprompt"],
                "summary": data["summary"]
            })
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {line.strip()} - {e}")

# Save the extracted data
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)

print(f"Extracted data saved to {output_file}")

```


then use the code here 

https://github.com/surenoobster/Generating_new_summaries/blob/main/Instruct_fine_tune_flanT5_category_2nd_approach.ipynb

the final type of dataset should look like this 

```
{
        "prompt": "Write highlights for this article for a high school student:\n\n",
        "input_text": "Education Minister Chris Pyne has been told to ‘man up’ by the Today Show’s Karl Stefanovic, who blasted ‘you can’t even get your policy through your own people’. Stefanovic stuck his teeth into Pyne a day after he grilled Prime Minister Tony Abbott about the axed carbon tax and federal budget, saying: ‘No one is buying what you are selling, what you are laying down. That is the problem.’ 'The reality here is that the budget position is in a shambles and every day it does get worse,’ he told the PM, as it was revealed Abbott’s popularity has fallen. Pyne appeared on the Today Show to discuss his unpopular Higher Education reforms as the government vowed it will launch a second attempt to pass them despite a Senate defeat on Tuesday. Scroll down for video . Today Show’s Karl Stefanovic told Education Minister Chris Pyne to 'man up' on Wednesday morning . Education minister Christopher Pyne later gave a press conference at Parliament House in Canberra . When Stefanovic pointed out ‘not even your own party has any faith in your arguments for existing reforms,’ Pyne argued: ’75 per cent of the budget is through.’ ‘Here’s the issue you can’t get your own policies through the Senate - and yes it’s hostile - but you can’t even get it through your own people,’ Karl laughed. ‘That’s not right Karl… I’m not in the least bit dismayed… my view is that’s democracy… this morning I am bouncing back and puting up a new reform bill,’ Pyne said. On Tuesday night Pyne was forced to deny that he ‘harassed’ senators in order to get them to support his higher education reforms, insisting that he holds an 'excellent relationship' with the Senate's crossbenchers. Pyne commented on the Government's higher education reform, after it was voted down in the Senate . PUP Senator Glenn Lazarus speaks in the Senate chamber at Parliament House in Canberra, Tuesday . Palmer United Party Senator Glenn Lazarus accused the Education Minister of pestering him and others in a statement, saying he'd been 'inundated with text messages from Christopher Pyne' despite having never given him his mobile number. But Mr Pyne dismissed the accusations, suggesting that the 'seven or eight' messages he'd sent Senator Lazarus were entirely acceptable and telling ABC's 7.30 Report that he would be 'happy to send him more'. Asking him about the allegations, Stefanovic jibed: ‘He’s a newly minted senator with virtually no experience, he was mocking you.’ Pyne vowed he will bounce back and propose a new version of the Higher Education reforms . He denied he was being blackmailed when Stefeanovic probed him about senators who did not vote for his reforms . ‘That’s a matter for Senator Lazarus. I intend to have a great relationship with him,’ Pyne replied. Announcing: ‘Today we’ll introduce a new higher education reform bill that picks up a lot of the issues that were raised by the crossbenchers, none by Labor or the Greens they are just saying “no” and are not part of the debate at all.’ ‘Whose fault is it that the bill was rejected?,’ Stefanovic asked. ‘Well the crossbenchers, the Labors and the Greens didn’t vote for it – four of the crossbenches did… great reform takes time, there is an inevitability about it because it’s great for our universities and students,’ Pyne said. Stefanovic continued his offence: ‘Ultimately you’ve got massive dramas getting any bill passed do you think that the senators are holding the country to ransom at the moment – are they blackmailing the country?’ The Prime Minister appeared on Channel Nine's Today show on Tuesday morning and faced a barrage of questions from host Karl Stefanovic over his controversial budget decisions . Avoiding answering the question, Pyne said: ‘As I see this, it is round two. As I said to Clive Palmer if it loses the first time it will be like losing the first test of a five test series.’ ‘They are blackmailing you in a way though aren’t they?,’ Stefanovic retorted. ‘I wouldn’t say that,’ Pyne said. ‘I think the PM said something like that yesterday “this is a form of blackmail”,’ Stefanovic reasoned. ‘I think he was saying… voting no in the Senate because you can’t have your way on one particular thing is not the way to negotiate with the government or with anybody,’ Pyne said. Stefanovic finished: ‘Why don’t you just double dissolution, man up and get on with it? ‘We’ve had a great year,’ Pyne replied. Education Minister Christopher Pyne has denied that he was 'harassing' Palmer United Party Senator Glenn Lazarus in order to gain his support for his higher education reforms package . The Abbott government will have another go at overhauling universities, defying defeat and bringing a new-look bill before parliament. Pyne will on Wednesday introduce a redrafted bill to deregulate universities, after the Senate knocked back the first attempt on Tuesday. Despite pleading with crossbench senators, the government failed to convince enough to jump on board the policy. The government will have to work on the Palmer United Party, Jacqui Lambie and Nick Xenophon to get the new bill through the Senate early next year. The redrafted bill will include other crossbench amendments that dump plans to increase interest rates on HECS and freeze interest charges while new parents take time out to raise kids. It will also include a scholarship fund targeted towards disadvantaged and rural students. On Tuesday Pyne also made an ironic slip when pronouncing the name of cross bencher Zhenya 'Dio' Wang while insisting he had an 'excellent relationship' with him - a mistake that was quickly pointed out by ABC reporter Leigh Sales. ABC journalist Leigh Sales, pointed Mr Pyne's ironic slip when pronouncing the name of cross bencher Zhenya 'Dio' Wang . 'Well some people pronounce it Wang, some people pronounce it Wong; it depends where you are on the spectrum. But if you wish to pick me up on that Leigh, that's a very small thing and I'm surprised you'd bother with it. Never the less, Dio and I are good friends and I will continue to try and get their support,' he replied to the quip. Clearly put-off by the slip, Mr Pyne then scolded Ms Sales, instructing her to study a bachelor of political science after being asked about the frequency of which he met with the crossbenchers and the quality of the relationships they shared. 'I have met with some of the crossbenchers many, many times – in fact I've met with some of them six or seven or eight times, but I'm not going to go through the day-to-day machinations of how government works, Leigh,' he said. 'You can go and study that at university if you wish to, in a bachelor of political science. The reality is that I'm working closely with the cross bench, I secured four of their votes today and I'll be back at it again tomorrow with a new reform bill.'",
        "summary": "Chris Pyne appeared on the Today Show on Wednesday morning .\nHe announced the government will launch a second attempt to pass education reforms after the senate rejected them on Tuesday .\nGlenn Lazarus has accused the Minister of 'harassing' him and other crossbenchers in order to gain their support for higher education reforms .\nMr Pyne defended himself by saying Mr Lazarus was the only crossbencher who had refused to meet with him for discussions, forcing him to text .\nStefanovic laid into Pyne a day after he blasted Tony Abbott about his unpopular budget .\nThe reforms were rejected by the Senate on Tuesday night with a vote of 33-31, with Senators Lazarus amongst those who voted against the bill ."
    },

```


and for Score instruct finetune 

```
    {
        "prompt": "Write highlights for this article with a Flesch-Kincaid score of 63:\n\n",
        "input_text": "Devon Still shared some amazing news about his daughter Leah on Wednesday. The Cincinnati Bengals star wrote that 4-year-old Leah, who has been bravely battling stage 4 Neuroblastoma for the past year, was told that her cancer is in remission. 'Today we received news from Leah's oncologist that her cancer, stage four neuroblastoma, is officially in REMISSION!' wrote Still on his Instagram. Scroll down for video . Cincinnati Bengals star Devon Still shared the heartwarming news on Wednesday that his daughter Leah's cancer was in remisison . Leah has been been battling stage 4 Neuroblastoma since last June, and captured the hearts of millions as she fights the disease . 'Today we received news from Leah's oncologist that her cancer, stage four neuroblastoma, is officially in REMISSION!' wrote Still on his Instagram . He went on to say; 'After 296 days of day dreaming about what it would feel like to hear the doctors say my daughter is in remission, I finally know the feeling. Funny thing is there is really no way of describing it because I never knew this feeling existed. When I look at my daughter all I can do is smile and hug her.' He then went on to thank everyone from the doctors who helped his daughter to those who wrote her notes and everyone who shared her story. Leah's journey has been followed by millions since she first began receiving treatment last June at the Children’s Hospital of Philadelphia. In September she had a tumor removed from her abdomen, and last month received what could be her final round of chemotherapy. She has also stayed remarkably active, walking in the Levi's Kids fashion show during New York Fashion Week in February and appearing in a music video for the song Truly Brave, a mashup of the Sara Bareilles song Brave and the Cyndi Lauper classic True Colors that was commissioned by Today anchor Hoda Kotb to raise funds for pediatric cancer research. In addition, she and her father have written a children's book to help other kids that are going through cancer treatment. Leah and her dad have also written a book that aims to help children who are battling the disease . The Bengals kept Still on their practice squad during the preseason so he could spend time with his daughter and receive their medical insurance . Leah's MBIG scans showed no active disease in her body, though the family is still waiting on an MRI and bone biopsy results . Last November, the Bengals presented a check for $1.3million to the Children's Hospital in Cincinnati for pediatric cancer treatment and research, money raised from the sale of Still jerseys. The team also honored Leah during a ceremony as well, and for the first time ever the young girl was actually able to attend a game since she had begun her treatment - and see her father play. The Bengals also helped Still and his daughter by excusing him from offseason activities so he could spend time with her in Philadelphia. And they kept him on the practice squad to start the season even though he was hurt so that he'd keep his medical coverage. June 2, 2014 and March 25, 2015 are days I will remember for the rest of my life. As everyone probably knows, June 2nd was the day doctors walked into the waiting room to tell me my daughter had cancer. It was the most devastating day of my life. March 25th , however, is feeling like the best day of my life. Today we received news from Leah's oncologist that her cancer, stage four neuroblastoma, is officially in REMISSION! After 296 days of day dreaming about what it would feel like to hear the doctors say my daughter is in remission, I finally know the feeling. Funny thing is there is really no way of describing it because I never knew this feeling existed. When I look at my daughter all I can do is smile and hug her. It was not easy but every day, and every treatment Leah fought like hell and kicked cancers butt! I'm so proud and blessed to call her my daughter. She has made an impact on me and on the world, at the age of four, that I can only wish to make in a lifetime. Thank you to my family and friends for the support through all those tough days. Thank you to everyone who has sent a letter to give Leah and our family motivation to keep fighting, a toy that helped Leah get through her days in the hospital, and more importantly a prayer that helped God hear our cries for healing. Thank you to the doctors at CHOP for putting together the best plan of action for my daughter. Thank you to Child Life members Sarah, Laura, and Lindsey for really turning what could be a scary place into a place where Leah would enjoy going because she knew she would have fun with you guys. Thank you to the Bengals for taking on my situation and standing by me and my family and for helping to raise money to fight pediatric cancer. To every media outlet and persons that helped raise much needed awareness, thank you. Leah is not done with treatments yet. She still needs more to make sure the cancer cells do not return and to build back up her immune system and other damage from the chemo but I know my little warrior will get through it!",
        "summary": "Cincinnati Bengals star Devon Still shared the heartwarming news on Wednesday that his daughter Leah's cancer is in remission .\nLeah has been been battling stage 4 Neuroblastoma since last June, and captured the hearts of millions as she fights the disease .\nStill said it was the best day of his life in a post on Instagram .\nThe Bengals kept Still on their practice squad during the preseason so he could spend time with his daughter and receive their medical insurance .\nThey also donated all profit from the sale of his jerseys to the Children's Hospital in Cincinnati, raising $1.3million .\nLeah and her dad have also written a book that aims to help children who are battling the disease ."
    }


```


