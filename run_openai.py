import os  
from langchain.llms import OpenAI  
from langchain.prompts import PromptTemplate  
  
# Set your OpenAI API key from the environment variable  
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  
  
def generate_text(prompt):  
    # Initialize the OpenAI LLM from LangChain  
    llm = OpenAI()  
      
    # Generate text based on the prompt  
    response = llm(prompt)  
    ai = response.response_metadata['model_name']
      
    return response, ai  
  
if __name__ == "__main__":  
    prompt = "Write a short story about a robot learning to be human."  
    generated_text, model = generate_text(prompt)  
      
    # Write the generated text to a file in the repository  
    with open("generated_story.txt", "w") as f:  
        f.write(model)  
