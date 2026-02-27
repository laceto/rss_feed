import os  
import polars as pl
import pandas as pd
from dotenv import load_dotenv, find_dotenv   
from langchain_core.documents import Document

def df_to_excel(
    df: pd.DataFrame, 
    path: str, 
    sheet: str = 'Sheet1'
    ) -> None:  
    """  
    Write a DataFrame to an Excel file.  
  
    Parameters:  
    df (pd.DataFrame): DataFrame to write to Excel.  
    path (str): Path of the Excel file to be created or overwritten.  
    sheet (str): Name of the sheet in the Excel file (default is 'Sheet1').  
  
    Returns:  
    None  
    """  
    try:  
        # Write the DataFrame to an Excel file  
        df.to_excel(path, sheet_name=sheet, index=False)  
        print(f"DataFrame successfully written to {path} in sheet '{sheet}'.")  
    except Exception as e:  
        print(f"An error occurred while writing to Excel: {e}")  


        
def get_file_paths(
    path: str, 
    file_pattern: str
    ) -> list:  
    """  
    Retrieve a list of all file paths in the specified directory and its subdirectories  
    that match the given file pattern.  
  
    This function scans the given directory (and its subdirectories) for files,  
    filters them based on the provided file pattern, and returns a list of their full paths.  
  
    Args:  
        path (str): Directory path where the search for files will be conducted.  
        file_pattern (str): File pattern to filter files (e.g., '.py' for Python files).  
  
    Returns:  
        list: List of full paths to files found in the specified directory that match the pattern.  
  
    Raises:  
        FileNotFoundError: If the specified directory does not exist.  
        Exception: For any other unexpected errors that may occur during file retrieval.  
    """  
    file_paths = []  
    try:  
        # List files in the base directory  
        # base_files = [os.path.join(path, f) for f in os.listdir(path)]  
          
        # Walk through the directory and subdirectories  
        for root, dirs, files in os.walk(path):  
            for file in files:  
                file_path = os.path.join(root, file)  
                file_paths.append(file_path)  
          
        # Filter for files matching the given pattern  
        filtered_file_paths = [file_path for file_path in file_paths if file_path.endswith(file_pattern)]  
        return filtered_file_paths  
  
    except FileNotFoundError as e:  
        print(f"Error: The directory '{path}' does not exist.")  
        raise e  
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")  
        raise e  

def print_env_file_contents() -> None:  
    """  
    Prints the contents of the .env file.  
  
    This function locates the .env file in the project directory, loads its   
    environment variables using the `dotenv` library, and prints each key-value   
    pair to the console.  
  
    Returns:  
        None: This function does not return any value.  
  
    Example:  
        print_env_file_contents()  
    """  
    try:  
        # Load environment variables from .env file  
        load_dotenv(find_dotenv())  
          
        # Get all environment variables  
        env_vars = os.environ  
  
        # Print each key-value pair in the .env file  
        for key, value in env_vars.items():  
            if key.startswith("DOTENV_"):  # Optional: filter if needed  
                print(f"{key}: {value}")  
            else:  
                print(f"{key}: {value}")  # Print all environment variables  
  
    except Exception as e:  
        print(f"Error printing .env file contents: {e}")  
        raise  

def initialize_token_provider() -> str:  
    """  
    Initializes the environment variables from a local .env file and retrieves a bearer token provider.  
  
    This function loads the environment variables from the `.env` file located in the project root directory.  
    It then creates a bearer token provider using Azure's `DefaultAzureCredential` for authentication   
    with Azure Cognitive Services.  
  
    Returns:  
        str: Bearer token provider for Azure Cognitive Services.  
  
    Raises:  
        Exception: If there is an error loading the environment variables or retrieving the token.  
  
    Example:  
        token_provider = initialize_token_provider()  
    """  
    try:  
        # Load environment variables from .env file  
        load_dotenv(find_dotenv())  
  
        # Create a bearer token provider for Azure Cognitive Services  
        token_provider = get_bearer_token_provider(  
            DefaultAzureCredential(),  
            "https://cognitiveservices.azure.com/.default"  
        )  
  
        return token_provider  
  
    except Exception as e:  
        print(f"Error initializing token provider: {e}")  
        raise  

def pl_read_excel(
    path: str, 
    sheet: str
    ) -> pl.DataFrame:  
    """  
    Reads an Excel file and returns the specified sheet as a Polars DataFrame.  
  
    This function leverages the Polars library to read data from an Excel file.   
    It retrieves the specified sheet and converts it into a Polars DataFrame for further processing.  
  
    Args:  
        path (str): Path to the Excel file to be read.  
        sheet (str): Name of the sheet in the Excel file to read.  
  
    Returns:  
        pl.DataFrame: Polars DataFrame containing the data from the specified sheet.  
  
    Raises:  
        FileNotFoundError: If the specified file does not exist.  
        ValueError: If the specified sheet name is not found in the Excel file.  
        Exception: For any other errors that may occur during the reading process.  
  
    Example:  
        df = pl_read_excel("data.xlsx", "Sheet1")  
    """  
    try:  
        return pl.read_excel(path, sheet_name=sheet)  
    except FileNotFoundError as e:  
        print(f"Error: The file '{path}' was not found. {e}")
        raise e  
    except ValueError as e:  
        print(f"Error: The sheet '{sheet}' was not found in the Excel file. {e}")  
        raise e
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")  
        raise e

def list_to_docs(
    docs: list[str]
    ) -> list[Document]:  
    """  
    Create a list of LangChain Document objects from a list of strings.  
  
    Args:  
        docs (list[str]): List of strings.  
  
    Returns:  
         -> list[Document]: List of LangChain Document objects.  
  
    Raises:  
        ValueError: If the input list is empty or contains non-string elements.  
    """  
    documents = []  
    try:  
        if not docs:  
            raise ValueError("The input list 'docs' cannot be empty.")  
          
        for string in docs:  
            if not isinstance(string, str):  
                raise ValueError("All elements in 'docs' must be strings.")  
                  
            # Create Document  
            doc = Document(page_content=string)  
            documents.append(doc)  
    except Exception as e:  
        print(f"An error occurred: {e}")  
      
    return documents  

def excel_to_docs(
    path: str, 
    sheet: str, 
    content_column: str, 
    metadata_columns: list[str] = None
    ) -> list[Document]:  
    """  
    Read an Excel file and convert rows to LangChain Document objects.  
  
    This function reads data from a specified Excel sheet and converts each row into a   
    LangChain Document. The specified content column is used as the main content of   
    the document, while optional metadata columns can be included as additional information.  
  
    Args:  
        path (str): Path to the Excel file to be read.  
        sheet (str): Name of the sheet in the Excel file to read.  
        content_column (str): Name of the column to use as document content.  
        metadata_columns (list[str], optional): List of columns to use as metadata for the documents. 
  
    Returns:  
         -> list[Document]: List of LangChain Document objects.  
  
    Raises:  
        ValueError: If the specified content column is not found in the Excel file.  
        Exception: For any other errors that may occur during the reading or processing of the Excel file.  
  
    Example:  
        documents = read_excel_to_documents("data.xlsx", "Sheet1", "Content", ["Author", "Date"])  
    """  
    try:  
        # Read the Excel file  
        df = pl_read_excel(path, sheet)  
        df = df.to_pandas()  
          
        # Validate content column exists  
        if content_column not in df.columns:  
            raise ValueError(f"Content column '{content_column}' not found in the Excel file")  
          
        # Prepare documents  
        documents = []  
        for index, row in df.iterrows():  
            # Prepare page content  
            page_content = str(row[content_column])  
              
            # Prepare metadata  
            metadata = {}  
            if metadata_columns:  
                for col in metadata_columns:  
                    if col in df.columns:  
                        # Convert value to string, handle NaN  
                        metadata[col] = str(row[col]) if pd.notna(row[col]) else "Unknown"  
              
            # Create Document  
            doc = Document(  
                page_content=page_content,  
                metadata=metadata  
            )  
            documents.append(doc)  
          
        return documents  
  
    except Exception as e:  
        print(f"Error reading Excel file: {e}")  
        return []

def flatten_list_of_lists(
    nested_list: list
    ) -> list: 
    """  
    Flattens a list of lists into a single list.  
  
    Args:  
        nested_list: A list containing lists to be flattened.  
  
    Returns:  
        A single list containing all elements from the nested lists.  
  
    Raises:  
        TypeError: If the input is not a list of lists.  
    """  
    try:  
        if not all(isinstance(i, list) for i in nested_list):  
            raise TypeError("Input must be a list of lists.")  
          
        return [item for sublist in nested_list for item in sublist]  
      
    except Exception as e:  
        print(f"An error occurred: {e}") 


def extract_attribute_doc(
    doc: Document, 
    attribute: str
    ) -> str:  
    """Extracts the specified attribute from a document object.  
  
    Args:  
        doc: LangChain Document object or an instance of a  from which to extract the attribute.  
        attribute: Name of the attribute to extract from the document.  
  
    Returns:  
        Value of the specified attribute.  
  
    Raises:  
        AttributeError: If the document does not have the specified attribute.  
    """  
    if not hasattr(doc, attribute):  
        raise AttributeError(f"Document does not have the '{attribute}' attribute.")  
    return getattr(doc, attribute)  

def extract_attribute_docs(
    docs: list[Document], 
    attribute
    ) -> list[str]:
    """Extracts the specified attribute from a document object or a list of document objects.  
  
    This function can handle both a single document object and a list of document objects  
    to extract the specified attribute.  
  
    Args:  
        docs: list[Document]: List of LangChain Document objects from which to extract the attribute.  
        attribute: Name of the attribute to extract from the document(s).  
  
    Returns:  
        List containing the values of the specified attribute from each document, or a single value  
        if a single document was provided.  
  
    Raises:  
        AttributeError: If a document does not have the specified attribute.  
        TypeError: If the input is not a document object or a list, or if the attribute is not a string.  
    """  
    try:  
        if isinstance(docs, list):  
            if not all(isinstance(doc, object) for doc in docs):  
                raise TypeError("All items in the list must be document objects.")  
        elif not isinstance(docs, object):  
            raise TypeError("Input must be a document object or a list of document objects.")  
  
        if not isinstance(attribute, str):  
            raise TypeError("Attribute name must be a string.")  
  
        attribute_values = []  
  
        # Extract attribute from a document using the helper function  
        for doc in (docs if isinstance(docs, list) else [docs]):  
            attribute_values.append(extract_attribute_doc(doc, attribute))  
        
        return attribute_values
  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return None  # Return None to indicate an error occurred  

  
def add_metadata_to_docs(
    docs: list[Document], 
    key: str, 
    value: any
    ) -> list[Document]:  
    """Update metadata of LangChain Document objects.  
  
    Given a list of LangChain Documents, this function returns a new list where each Document's metadata  
    is updated to include a specified key-value pair.  
  
    Args:  
        docs (list[Document]): List of LangChain Document objects.  
        key (str): Key to be added to the metadata.  
        value (any): Value associated with the key to be added to the metadata.  
  
    Returns:  
        list[Document]: List of LangChain Document objects with updated metadata.  
    """  
    new_docs = []  
    try:  
        for doc in docs:  
            new_metadata = dict(doc.metadata or {})  # Copy existing metadata or use an empty dict  
            new_metadata[key] = value  
            new_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))  
        return new_docs  
  
    except Exception as e:  
        print(f"An error occurred while adding metadata: {e}")  
        return []  
    

def df_to_docs(
    df: pd.DataFrame, 
    content_column: str, 
    metadata_columns: list[str] = None
    ) -> list[Document]:  
    """  
    Prepare a list of Document objects from a DataFrame.  
  
    Args:  
        df (pd.DataFrame): The DataFrame containing the data.  
        content_column (str): The name of the column containing the content for the documents.  
        metadata_columns (list[str], optional): A list of columns to use as metadata for the documents. Defaults to None.  
  
    Returns:  
        list[Document]: A list of Document objects created from the DataFrame rows.  
  
    Raises:  
        ValueError: If the content_column is not found in the DataFrame.  
    """  
    documents = []  
    try:  
        # Validate content column exists  
        if content_column not in df.columns:  
            raise ValueError(f"Content column '{content_column}' not found in the DataFrame")  
          
        # Prepare documents  
        for index, row in df.iterrows():  
            # Prepare page content  
            page_content = str(row[content_column])  
              
            # Prepare metadata  
            metadata = {}  
            if metadata_columns:  
                for col in metadata_columns:  
                    if col in df.columns:  
                        # Convert value to string, handle NaN  
                        metadata[col] = str(row[col]) if pd.notna(row[col]) else "Unknown"  
             
            # Create Document  
            doc = Document(  
                page_content=page_content,  
                metadata=metadata  
            )  
            documents.append(doc)  
    except Exception as e:  
        print("An error occurred:")
        raise e  
      
    return documents  
