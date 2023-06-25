import os
import pandas as pd
import nbformat
import chardet
import streamlit as st
from nbconvert import PythonExporter
from github import Github
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.vectorstores import FAISS
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere


# OPENAI_API_KEY = os.environ.get("OPENAI_API")
GIT_TOKEN = os.environ.get("GIT_TOKEN")
COHERE_API_KEY = "zb0OV3gUQKzYO72fRdiNfD4CGB4czVmTIkWxLTCX"
GPT_TOKEN_LIMIT = 2048
MAX_LINE_LENGTH = 80


def preprocess_file(file_path):
    print(f"Processing file '{file_path}'")
    code_extensions = [".py", ".java", ".cpp", ".js", ".c", ".html", ".css", ".rb"]  # Add more extensions as needed
    chunks = []
    
    if file_path.endswith(".ipynb"):
        # Process Jupyter Notebook file
        try:
            notebook = nbformat.read(file_path, as_version=4)
            exporter = PythonExporter()
            
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    code = exporter.source_from_cell(cell)
                   # Split long lines of code into multiple lines
                    code_lines = []
                    for line in code.split("\n"):
                        if len(line) > MAX_LINE_LENGTH:
                            # Split the line into multiple lines
                            line_parts = [line[i:i+MAX_LINE_LENGTH] for i in range(0, len(line), MAX_LINE_LENGTH)]
                            code_lines.extend(line_parts)
                        else:
                            code_lines.append(line)
                    # Add the lines to the chunks list
                    chunks.extend(code_lines)
        except Exception as e:
            print(f"Error processing Jupyter Notebook file '{file_path}': {str(e)}")
    
    elif any(file_path.endswith(ext) for ext in code_extensions):
        # Process other code files
        try:
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read())
                encoding = result["encoding"]

                with open(file_path, "r", encoding=encoding) as f:
                    for line in f:
                        if len(line) > MAX_LINE_LENGTH:
                        # Split the line into multiple lines
                            line_parts = [line[i:i+MAX_LINE_LENGTH] for i in range(0, len(line),
                                            MAX_LINE_LENGTH)]
                            chunks.extend(line_parts)
                        else:
                            chunks.append(line)
        except Exception as e:
            print(f"Error processing code file '{file_path}': {str(e)}")
            pass
    return chunks

def fetch_github_repos(username):
    client = Github(GIT_TOKEN)
    user = client.get_user(username)
    repos = user.get_repos()

    repo_info = []
    try:
        for repo in repos:
            try:
                contents = repo.get_contents("")
                code_contents = []
                for content in contents:
                    file_path = content.path
                    chunks = preprocess_file(file_path)
                    code_contents.extend(chunks)

                repo_info.append({
                    "name": repo.name,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "labels": [label.name for label in repo.get_labels()],
                    "issues": repo.get_issues(state="all"),
                    "contents": code_contents,
                })
            except Exception as e:
                print(f"Error fetching contents of repository '{repo.name}': {str(e)}")
                continue
    except Exception as e:
        print(f"Error fetching repositories: {str(e)}")
    return repo_info


def analyze_repos(repo_info):
    repo_info_df = pd.DataFrame(repo_info)
    repo_info_df.to_csv("repo_data.csv")

    loader = CSVLoader(file_path="repo_data.csv", encoding="utf-8")
    csv_data = loader.load()
    csv_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    vectors = FAISS.from_documents(csv_data, csv_embeddings)

    context = """
    You are a Supersmart Github Repository AI system. You are a superintelligent AI that answers questions about Github Repositories and can understand the technical complexity of a repo.

    You have been asked to find the most technically complex and challenging repository from the given CSV file.

    To measure the technical complexity of a GitHub repository, you will analyze various factors such as the number of commits, branches, pull requests, issues, contents, number of forks, stars, and contributors. 
    Additionally, you will consider the programming languages used, the size of the codebase, and the frequency of updates.

    Calculate the complexity score for each project by assigning weights to each factor and summing up the weighted scores. The project with the highest complexity score will be considered the most technically complex.
    
    Analyze the following factors to determine the technical complexity of the codebase:

    1. Description
    2. Languages used in the repository
    3. Number of stars
    4. Number of forks
    5. Labels of the repository
    6. Issues associated with the repository
    7. Contents of the repository

    You can consider other factors as well if you think they are relevant for determining the technical complexity of a GitHub repository.

    Please provide a detailed analysis to justify your selection of the most technically complex repository.
    """

    prompt_template = """
    Understand the following to answer the question in an efficient way:

    {context}

    Question: {question}

    Now answer the question. Let's think step by step:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0),
                                        chain_type="stuff",
                                        retriever=vectors.as_retriever(),
                                        input_key="question",
                                        chain_type_kwargs=chain_type_kwargs)

    query = """
    Which is the most technically challenging repository from the given CSV file?

    Return
    the name of the repository, 
    the link to the repository, 
    and the analysis of the repository showing why it is the most technically challenging/complex repository. 
    Provide a detailed analysis to support your answer.

    The output should be in the following format:

    Repository Name: <name of the repository>
    Repository Link: <link to the repository>
    Analysis: <analysis of the repository>

    Provide a clickable link to the repository as well like this:
    [Repository Name](Repository Link)
    """

    result = chain({"question": query})
    return result


def main():
    st.set_page_config(page_title="GitHub Repository Analyzer")
    st.title("Github Repository Analysis")

    # Get GitHub username from user input
    username = st.text_input("Enter a GitHub username to analyze:", value="")

    if username:
        # Fetch GitHub repositories and analyze them
        repo_info = fetch_github_repos(username)
        result = analyze_repos(repo_info)

        # Display the results
        st.header("Analysis Results")
        st.markdown(f"**GitHub Username:** {username}")
        st.markdown(f"**Most Technically Challenging Repository:** {result['name']}")
        st.markdown(f"**Repository Link:** [{result['name']}]({result['url']})")
        st.markdown(f"**Analysis:** {result['analysis']}")


if __name__ == '__main__':
    main()
