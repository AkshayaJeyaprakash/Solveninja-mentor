import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv


class RAG:
    def __init__(self):
        """
        Initialize the VectorDB (FAISS) with the embedding model and OpenAI client in order to access the GPT-4o-mini model.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.embedding_model = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=faiss.IndexFlatL2(1536),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.prompt_path = "prompt.pmt"

    def indexing_pipeline(self, text: str, metadata: dict = None):
        """
        Add a document to the FAISS vector database.
        :param text: The document content to add
        :param metadata: Additional metadata for the document (optional)
        """
        document = Document(page_content=text, metadata=metadata or {})
        self.vector_store.add_documents([document])

    def retrieve_document(self, query: str):
        """
        Perform a similarity search using the provided query.
        :param query: The query string
        :return: The response from the model after retrieval
        """
        return self.vector_store.similarity_search(query)

    def augment_prompt(self, question: str, context: str) -> str:
        """
        Loads the prompt template from a file and populates it with the given question and context.
        :param prompt_path: Path to the prompt template file (.pmt).
        :param question: The problem statement provided by the student.
        :param context: The past success story fetched from RAG.
        :return: The fully formatted prompt ready to send to the model.
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as file:
                template = file.read()
            formatted_prompt = template.format(question=question, context=context)
            return formatted_prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
        except KeyError as e:
            raise ValueError(f"Missing placeholder in prompt template: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the prompt: {e}")

    def generate_response(self, prompt: str):
        """
        Generate a response from the GPT-4o model using the provided query and context data.
        :param query: The query string provided by the user
        :param data: The context retrieved from the vector database
        :return: The model's generated response
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return response.choices[0].message.content

    def rag_pipeline(self, query : str):
        """
        Complete the RAG (Retrieval-Augmented Generation) process using the provided user query.
        :param query: The input query string from the user.
        :return: The model's generated response after retrieving relevant context and augmenting the prompt.
        """
        context = self.retrieve_document(query)
        prompt = self.augment_prompt(query, context)
        response = self.generate_response(prompt)
        return response