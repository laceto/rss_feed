# import os  
# from langchain_openai import OpenAI 
# from langchain.prompts import PromptTemplate  
  
# # Set your OpenAI API key from the environment variable  
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  
  
# def generate_text(prompt):  
#     # Initialize the OpenAI LLM from LangChain  
#     llm = OpenAI(model="gpt-4.1-nano", temperature=0)  
    
      
#     # Generate text based on the prompt  
#     response = llm(prompt)  
#     # ai = response.response_metadata['model_name']
      
#     return response  
  
# if __name__ == "__main__":  
#     prompt = "Write a short story about a robot learning to be an actuary."  
#     generated_text = generate_text(prompt)  
      
#     # Write the generated text to a file in the repository  
#     with open("generated_story.txt", "w") as f:  
#         f.write(generated_text)  


# # Open the file in read mode  
# with open('output/feeds2025-07-31.txt', 'r') as file:  
#     # Read the entire file  
#     contents = file.read()  
  
# # Write the generated text to a file in the repository  
# with open("feeds2025-07-31.txt", "w") as f:
#     f.write(contents) 

# import pandas as pd

# df = pd.read_csv('output/feeds2025-07-31.txt', sep='\t')
# print(df)

from utils import *

file_list = get_file_paths('output', file_pattern='txt')

dfs = [pd.read_csv(file, sep='\t') for file in file_list]
dfs = pd.concat(dfs, ignore_index=True)

dfs['description'] = dfs['title'] + dfs['description']


print(dfs.columns)