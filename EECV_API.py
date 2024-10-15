// use EECV API #7
// I have tried to fixed the issue you have mentioned with the use of Python and OpenAI
// Before using the code please kindly install the below mentioned thing in the terminal for usage of the program

// pip install openai


import openai

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key_here'

# List of ECCV paper titles
paper_titles = [
    "Learning Robust Representations for Image Segmentation",
    "Generative Models for Object Detection",
    "Attention Mechanisms in Deep Learning for Computer Vision"
]

# Function to summarize each paper title
def summarize_paper(title):
    prompt = f"Summarize the research topic and potential contributions of the paper titled '{title}' in simple terms."

    # Call the OpenAI API to generate a summary
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.6
    )

    summary = response.choices[0].text.strip()
    return summary

# Loop through each title and print its summary
for title in paper_titles:
    summary = summarize_paper(title)
    print(f"Paper Title: {title}\nSummary: {summary}\n")
