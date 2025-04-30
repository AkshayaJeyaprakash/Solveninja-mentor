import os
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import Optional
from cachetools import TTLCache
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        """
        Initialize the VectorDB (FAISS) with the embedding model and OpenAI client in order to access the GPT-4o-mini model.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        logger.info("Initializing embedding model and OpenAI client.")
        self.embedding_model = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=faiss.IndexFlatL2(1536),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.prompt_path = "prompt.pmt"
        self.session_manager = TTLCache(maxsize=250, ttl=3600)
        

    def indexing_pipeline(self, text: str, metadata: dict = None):
        """
        Add a document to the FAISS vector database.
        :param text: The document content to add
        :param metadata: Additional metadata for the document (optional)
        """
        document = Document(page_content=text, metadata=metadata or {})
        try:
            document_id = self.vector_store.add_documents([document])
            logger.info("Document indexed successfully with metadata: %s", metadata)
            return document_id
        except Exception as e:
            logger.exception("Failed to index document: %s", e)
            raise

    def delete_document(self, id: str):
        """
        Delete a document from the FAISS vector database.
        :param id: The document ID to delete
        """
        try:
            if id not in self.vector_store.docstore._dict:
                raise ValueError(f"Document with ID {id} not found.")
            self.vector_store.delete([id])
            logger.info("Document deleted successfully with id: %s", id)
        except Exception as e:
            logger.exception("Unexpected error while deleting document: %s", e)
            raise


    def retrieve_document(self, query: str):
        """
        Perform a similarity search using the provided query.
        :param query: The query string
        :return: The response from the model after retrieval
        """
        try:
            logger.debug("Retrieving document for query: '%s'", query)
            result = self.vector_store.similarity_search(query)
            logger.debug("Retrieved similar documents.")
            return result[0].page_content
        except Exception as e:
            logger.exception("Error retrieving document: %s", e)
            raise

    def augment_prompt(self, question: str, context: str):
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
            logger.info("Prompt augmented successfully.")
            return formatted_prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
        except KeyError as e:
            raise ValueError(f"Missing placeholder in prompt template: {e}")
        except Exception as e:
            raise SystemError(f"An error occurred while loading the prompt: {e}")

    def generate_response(self, prompt: str, session_id: str):
        """
        Generate a response from the GPT-4o model using the provided query and context data.
        :param query: The query string provided by the user
        :param data: The context retrieved from the vector database
        :return: The model's generated response
        """
        if session_id != "":
            history = self.session_manager[session_id]
            messages = history + [{"role": "user", "content": prompt}]
        else:
            session_id = str(uuid.uuid4())
            self.session_manager[session_id] = []
            messages = [{"role": "user", "content": prompt}]

        logger.info(messages)
        try:
            logger.info("Sending prompt to OpenAI model for completion.")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000
            )
        except Exception as e:
            logger.exception("Failed to fetch response from OpenAI: %s", e)
            raise

        logger.info("Response generated successfully.")
        assistant_reply = response.choices[0].message.content
        self.session_manager[session_id].append({"role": "user", "content": prompt})
        self.session_manager[session_id].append({"role": "assistant", "content": assistant_reply})
        return assistant_reply, session_id

    def rag_pipeline(self, query: str, session_id: Optional[str] = None):
        """
        Complete the RAG (Retrieval-Augmented Generation) process using the provided user query.
        :param query: The input query string from the user.
        :return: The model's generated response after retrieving relevant context and augmenting the prompt.
        """
        if session_id is not None:
            response = self.generate_response(query, session_id)
            return response
        context = self.retrieve_document(query)
        prompt = self.augment_prompt(query, context)
        response, session_id = self.generate_response(prompt, "")
        return response, session_id