import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('./logs/functions_for_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY") or ""
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY") or ""
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY","")

# os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY") or ""
ollama_api_key = os.environ.get("OLLAMA_API_KEY")
nvidia_key = os.environ.get("NVIDIA_API_KEY")
if not nvidia_key:
    raise ValueError("NVIDIA_API_KEY environment variable is missing!")
if not ollama_api_key:
    raise ValueError("OLLAMA_API_KEY environment variable is missing!")

# ============================================================================
# PHASE 1: Module-level singleton caches for performance optimization
# ============================================================================
import threading

_GLOBAL_EMBEDDINGS = None
_GLOBAL_VECTOR_STORES = None
_LLM_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_embeddings():
    """Get cached OllamaEmbeddings instance. Initializes on first call only."""
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        with _CACHE_LOCK:
            if _GLOBAL_EMBEDDINGS is None:
                logger.info("Initializing OllamaEmbeddings (first call only)")
                _GLOBAL_EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    return _GLOBAL_EMBEDDINGS


def get_vector_stores():
    """Get cached vector stores. Loads from disk on first call only."""
    global _GLOBAL_VECTOR_STORES
    if _GLOBAL_VECTOR_STORES is None:
        with _CACHE_LOCK:
            if _GLOBAL_VECTOR_STORES is None:
                logger.info("Loading vector stores (first call only)")
                embeddings = get_embeddings()
                _GLOBAL_VECTOR_STORES = {
                    'chunks': FAISS.load_local("chunks_vector_store", embeddings, allow_dangerous_deserialization=True),
                    'summaries': FAISS.load_local("chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True),
                    'quotes': FAISS.load_local("book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True),
                }
                logger.info("Vector stores loaded successfully")
    return _GLOBAL_VECTOR_STORES


def get_retrievers():
    """Get retriever objects. Uses cached vector stores."""
    vs = get_vector_stores()
    return {
        'chunks': vs['chunks'].as_retriever(search_kwargs={"k": 3}),
        'summaries': vs['summaries'].as_retriever(search_kwargs={"k": 3}),
        'quotes': vs['quotes'].as_retriever(search_kwargs={"k": 3}),
    }


def get_llm(model="minimaxai/minimax-m3", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1") -> ChatNVIDIA:
    """Get cached ChatNVIDIA instance. Creates once per unique model/temperature combo."""
    global _LLM_CACHE
    cache_key = f"{model}_{temperature}"
    if cache_key not in _LLM_CACHE:
        with _CACHE_LOCK:
            if cache_key not in _LLM_CACHE:
                logger.info(f"Creating ChatNVIDIA instance for {model} (first call only)")
                _LLM_CACHE[cache_key] = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it"
            )
    return _LLM_CACHE[cache_key]


class EmbeddingCache:
    """Thread-safe cache for question embeddings."""
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def embed_query(self, text):
        """Get cached embedding or generate if not cached."""
        text_hash = hash(text)
        with self.lock:
            if text_hash not in self.cache:
                logger.debug(f"Embedding query (cache miss): {text[:50]}")
                self.cache[text_hash] = get_embeddings().embed_query(text)
            else:
                logger.debug(f"Using cached embedding (cache hit)")
        return self.cache[text_hash]


# ============================================================================
# END PHASE 1
# ============================================================================

# ============================================================================
# PHASE 2: Parallelization and confidence-based early exit
# ============================================================================
import asyncio

async def retrieve_all_contexts_parallel(question):
    """Fetch all 3 retrieval sources in parallel using asyncio. Returns dict with all contexts."""
    retrievers = get_retrievers()
    try:
        logger.debug("Starting parallel retrieval of all 3 sources")
        results = await asyncio.gather(
            asyncio.to_thread(retrievers['chunks'].invoke, question),
            asyncio.to_thread(retrievers['summaries'].invoke, question),
            asyncio.to_thread(retrievers['quotes'].invoke, question),
        )
        logger.info("Parallel retrieval completed")
        return {
            'chunks': results[0],
            'summaries': results[1],
            'quotes': results[2]
        }
    except Exception as e:
        logger.error(f"Error in parallel retrieval: {str(e)}", exc_info=True)
        raise


def evaluate_answer_confidence(answer_text: str, context: str) -> float:
    """Evaluate confidence score of answer based on context presence. Returns 0.0-1.0."""
    if not answer_text or not context:
        return 0.0

    answer_lower = answer_text.lower()
    context_lower = context.lower()

    words = answer_lower.split()
    matched_words = sum(1 for word in words if word in context_lower)
    confidence = min(1.0, matched_words / max(len(words), 1) * 0.9)

    logger.debug(f"Answer confidence: {confidence:.2f} ({matched_words}/{len(words)} words in context)")
    return confidence


def check_grounding(answer_text: str, context: str) -> bool:
    """Check if answer is grounded in context. Returns boolean."""
    if not answer_text or not context:
        return False

    answer_lower = answer_text.lower()
    context_lower = context.lower()

    words = answer_lower.split()
    grounding_threshold = 0.3
    matched_words = sum(1 for word in words if len(word) > 3 and word in context_lower)

    is_grounded = matched_words / max(len([w for w in words if len(w) > 3]), 1) > grounding_threshold
    logger.debug(f"Answer grounding: {is_grounded}")
    return is_grounded


# ============================================================================
# END PHASE 2
# ============================================================================

# ============================================================================
# PHASE 3: Advanced optimizations (fast embeddings, model selection, caching)
# ============================================================================
from time import time
from functools import lru_cache

_FAST_EMBEDDINGS = None


def get_fast_embeddings():
    """Get cached fast embeddings using sentence-transformers. Much faster than OllamaEmbeddings."""
    global _FAST_EMBEDDINGS
    if _FAST_EMBEDDINGS is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Initializing fast embeddings with sentence-transformers (first call only)")
            _FAST_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"Fast embeddings unavailable: {e}, falling back to OllamaEmbeddings")
            _FAST_EMBEDDINGS = get_embeddings()

    return _FAST_EMBEDDINGS


class RequestCache:
    """Thread-safe LRU cache for question embeddings and responses. Phase 3 optimization."""

    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()

    def get_cached_response(self, question_hash: int, similarity_threshold: float = 0.95):
        """Check if similar question was answered recently."""
        with self.lock:
            current_time = time()

            for cached_hash, (response, timestamp) in list(self.cache.items()):
                if current_time - timestamp > self.ttl_seconds:
                    del self.cache[cached_hash]
                    del self.access_times[cached_hash]
                    continue

                if abs(cached_hash - question_hash) / max(abs(cached_hash), 1) < (1 - similarity_threshold):
                    logger.debug(f"Cache hit for similar question")
                    self.access_times[cached_hash] = current_time
                    return response

        return None

    def cache_response(self, question_hash: int, response: str):
        """Cache a response for a question."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
                logger.debug(f"Evicted LRU cached entry")

            self.cache[question_hash] = (response, time())
            self.access_times[question_hash] = time()


_REQUEST_CACHE = RequestCache()


# ============================================================================
# END PHASE 3
# ============================================================================

def create_retrievers():
    """Create and return retrievers from vector stores."""
    try:
        logger.debug("Starting create_retrievers")
        embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        logger.debug("Created OllamaEmbeddings")

        chunks_vector_store = FAISS.load_local("chunks_vector_store", embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded chunks_vector_store")

        chapters_vector_store = FAISS.load_local("chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded chapter_summaries_vector_store")

        book_quotes_vector_store = FAISS.load_local("book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded book_quotes_vectorstore")

        chunks_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 3})
        chapters_retriever = chapters_vector_store.as_retriever(search_kwargs={"k": 3})
        book_quotes_retriever = book_quotes_vector_store.as_retriever(search_kwargs={"k": 3})

        logger.info("Successfully created all retrievers")
        return chunks_retriever, chapters_retriever, book_quotes_retriever
    except FileNotFoundError as e:
        logger.error(f"Vector store file not found: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in create_retrievers: {str(e)}", exc_info=True)
        raise
    
def retrieve_context_per_question(state:dict):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    try:
        logger.debug(f"retrieve_context_per_question called with question: {state.get('question', 'N/A')[:50]}")
        question = state['question']
        retrievers = get_retrievers()
        chunks_retriever = retrievers['chunks']
        chapters_retriever = retrievers['summaries']
        book_quotes_retriever = retrievers['quotes']

        # Retrieve relevant documents
        logger.info("Retrieving relevant chunks...")
        print("Retrieving relevant chunks...")
        docs = chunks_retriever.invoke(question)
        logger.debug(f"Retrieved {len(docs)} chunk documents")
        context = " ".join(doc.page_content for doc in docs)

        logger.info("Retrieving relevant chapter summaries...")
        print("Retrieving relevant chapter summaries...")
        docs_summaries = chapters_retriever.invoke(state["question"])
        logger.debug(f"Retrieved {len(docs_summaries)} summary documents")
        context_summaries = " ".join(f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries)

        logger.info("Retrieving relevant book quotes...")
        print("Retrieving relevant book quotes...")
        docs_quotes = book_quotes_retriever.invoke(state["question"])
        logger.debug(f"Retrieved {len(docs_quotes)} quote documents")
        context_quotes = " ".join(doc.page_content for doc in docs_quotes)

        all_contexts = context + context_summaries + context_quotes
        logger.info(f"Total context retrieved: {len(all_contexts)} characters")
        return {"context": all_contexts, "question": question}
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in retrieve_context_per_question: {str(e)}", exc_info=True)
        raise

def keep_only_relevant_context_chain():
    only_relevant_context_prompt_template = """
    you receive a query :{query} and retrieved documents :{retrieved_docs} from the vector store. The retrieved documents may contain irrelevant information. Your task is to extract only the relevant information(just to filter out the non relevant information.) from the retrieved documents that can help answer the query :{query}. Do not add any information that is not present in the retrieved documents. Just extract the relevant information from the retrieved documents and return it."""
    class KeepReleventContext(BaseModel):
        relevant_content:str = Field(description="The relevant content from the retrieved documents that is relevant to the query.")
    
    keep_only_relevant_context_prompt = PromptTemplate(template=only_relevant_context_prompt_template, input_variables=["query", "retrieved_docs"])
    keep_only_relevent_context_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)   
    only_relevant_context_chain = keep_only_relevant_context_prompt | keep_only_relevent_context_llm.with_structured_output(KeepReleventContext, method="json_schema")
    return only_relevant_context_chain
    
    
def keep_only_relevant_content(state):
    """
    Keeps only the relevant content from the retrieved documents that is relevant to the query.

    Args:
    question: The query question.
    context: The retrieved documents.
    chain: The LLMChain instance.

    Returns:
    The relevant content from the retrieved documents that is relevant to the query.
    """
    try:
        logger.debug(f"keep_only_relevant_content called")
        question = state["question"]
        context = state["context"]
        logger.debug(f"Question length: {len(question)}, Context length: {len(context)}")

        input_data = {
            "query": question,
            "retrieved_docs": context
        }
        logger.info("Creating keep_only_relevant_context_chain...")
        print("keeping only the relevant content...")
        print("--------------------")
        only_relevant_context_chain = keep_only_relevant_context_chain()
        logger.debug("Chain created successfully")

        logger.info("Invoking chain to filter relevant content...")
        output = only_relevant_context_chain.invoke(input_data)
        logger.debug("Chain invocation completed")

        relevant_content = output.relevant_content
        relevant_content = "".join(relevant_content)
        logger.info(f"Filtered relevant content: {len(relevant_content)} characters")
        # relevant_content = escape_quotes(relevant_content)

        return {"relevant_context": relevant_content, "context": context, "question": question}
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in keep_only_relevant_content: {str(e)}", exc_info=True)
        raise

def build_questions_using_chain_of_thoughts_chain():
    class QuestionAnswerFromContext(BaseModel):
        answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")
    question_answer_from_context_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)


    question_answer_cot_prompt_template = """ 
    Examples of Chain-of-Thought Reasoning

    Example 1

    Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
    Question: Who is the tallest person?
    Reasoning Chain:
    The context tells us Mary is taller than Jane
    It also says Jane is shorter than Tom
    And Tom is the same height as David
    So the order from tallest to shortest is: Mary, Tom/David, Jane
    Therefore, Mary must be the tallest person

    Example 2
    Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
    A third spell created a bright light at the end of the caster's wand.
    Question: Based on the context, if Harry cast these spells, what could he do?
    Reasoning Chain:
    The context describes three different magic spells
    The first spell allows turning a person into an animal temporarily
    The second spell can levitate or float objects
    The third spell creates a bright light
    If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
    So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
    Instructions.

    Example 3 
    Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
    Question: Why did Harry receive a broomstick for his birthday?
    Reasoning Chain:
    The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
    However, the context does not provide any information about why he received that specific present or who gave it to him.
    There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
    Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

    For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
    just like in the previous examples.
    Context
    {context}
    Question
    {question}
    """

    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )
    question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext, method="json_schema")
    return question_answer_from_context_cot_chain

def generate_answer_from_context(state):
    """
    Answers a question from a given context.

    Args:
        question: The query question.
        context: The context to answer the question from.
        chain: The LLMChain instance.

    Returns:
        The answer to the question from the context.
    """
    try:
        logger.debug(f"generate_answer_from_context called")
        question = state["question"]
        context = state["relevant_context"]
        logger.debug(f"Question: {question[:50]}, Context length: {len(context)}")

        input_data = {
            "question": question,
            "context": context
        }
        logger.info("Building chain of thoughts chain...")
        questions_using_chain_of_thoughts_chain = build_questions_using_chain_of_thoughts_chain()
        logger.debug("Chain built successfully")

        print("Answering the question from the retrieved context...")
        logger.info("Invoking chain to generate answer...")
        output = questions_using_chain_of_thoughts_chain.invoke(input_data)
        logger.debug("Chain invocation completed")

        answer = output.answer_based_on_content
        logger.info(f"Generated answer: {answer[:100]}...")
        print(f'answer before checking hallucination: {answer}')

        # Phase 2: Add confidence and grounding scoring
        confidence = evaluate_answer_confidence(answer, context)
        is_grounded = check_grounding(answer, context)

        return {
            "answer": answer,
            "context": context,
            "question": question,
            "confidence": confidence,
            "is_grounded": is_grounded,
            "replan_count": 0
        }
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in generate_answer_from_context: {str(e)}", exc_info=True)
        raise
    
    
def build_is_relevant_content_chain():
    is_relevant_content_prompt_template = """you receive a query: {query} and a context: {context} retrieved from a vector store.
    You need to determine if the document is relevant to the query."""

    class Relevance(BaseModel):
        is_relevant: bool = Field(description="Whether the document is relevant to the query.")
        explanation: str = Field(description="An explanation of why the document is relevant or not.")

    # is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)
    # is_relevant_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)
    is_relevant_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)

    is_relevant_content_prompt = PromptTemplate(
        template=is_relevant_content_prompt_template,
        input_variables=["query", "context"],
        # partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()},
    )
    is_relevant_content_chain = is_relevant_content_prompt | is_relevant_llm.with_structured_output(Relevance, method="json_schema")
    return is_relevant_content_chain

def is_relevant_content(state):
    """
    Determines if the document is relevant to the query.

    Args:
        question: The query question.
        context: The context to determine relevance.
    """
    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "context": context
     }
    is_relevant_content_chain = build_is_relevant_content_chain()

    # Invoke the chain to determine if the document is relevant
    output = is_relevant_content_chain.invoke(input_data)
    print("Determining if the document is relevant...")
    if output["is_relevant"] == True:
        print("The document is relevant.")
        return "relevant"
    else:
        print("The document is not relevant.")
        return "not relevant"
    
def create_is_grounded_on_facts_chain():
    class is_grounded_on_facts(BaseModel):
        """
        Output schema for the rewritten question.
        """
        grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    is_grounded_on_facts_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)
    is_grounded_on_facts_prompt_template = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
    you don't mind if it doesn't make sense, as long as it is grounded in the context.
    output a json containing the answer to the question, and appart from the json format don't output any additional text.

    """
    is_grounded_on_facts_prompt = PromptTemplate(
        template=is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )
    is_grounded_on_facts_chain = is_grounded_on_facts_prompt | is_grounded_on_facts_llm.with_structured_output(is_grounded_on_facts, method="json_schema")
    return is_grounded_on_facts_chain

def create_can_be_answered_chain():
    can_be_answered_prompt_template = """You receive a query: {question} and a context: {context}. 
    You need to determine if the question can be fully answered based on the context."""

    class QuestionAnswer(BaseModel):
        can_be_answered: bool = Field(description="binary result of whether the question can be fully answered or not")
        explanation: str = Field(description="An explanation of why the question can be fully answered or not.")

    # can_be_answered_json_parser = JsonOutputParser(pydantic_object=QuestionAnswer)

    answer_question_prompt = PromptTemplate(
        template=can_be_answered_prompt_template,
        input_variables=["question","context"],
        # partial_variables={"format_instructions": can_be_answered_json_parser.get_format_instructions()},
    )

    # can_be_answered_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)
    can_be_answered_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)
    can_be_answered_chain = answer_question_prompt | can_be_answered_llm.with_structured_output(QuestionAnswer, method="json_schema")
    return can_be_answered_chain

def create_is_distilled_content_grounded_on_content_chain():
    is_distilled_content_grounded_on_content_prompt_template = """you receive some distilled content: {distilled_content} and the original context: {original_context}.
        you need to determine if the distilled content is grounded on the original context.
        if the distilled content is grounded on the original context, set the grounded field to true.
        if the distilled content is not grounded on the original context, set the grounded field to false."""
    
    class IsDistilledContentGroundedOnContent(BaseModel):
        grounded: bool = Field(description="Whether the distilled content is grounded on the original context.")
        explanation: str = Field(description="An explanation of why the distilled content is or is not grounded on the original context.")

    #is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(pydantic_object=IsDistilledContentGroundedOnContent)

    is_distilled_content_grounded_on_content_prompt = PromptTemplate(
        template=is_distilled_content_grounded_on_content_prompt_template,
        input_variables=["distilled_content", "original_context"],
        # partial_variables={"format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()},
    )

    # is_distilled_content_grounded_on_content_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)
    is_distilled_content_grounded_on_content_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)

    is_distilled_content_grounded_on_content_chain = is_distilled_content_grounded_on_content_prompt | is_distilled_content_grounded_on_content_llm.with_structured_output(IsDistilledContentGroundedOnContent)
    return is_distilled_content_grounded_on_content_chain

is_distilled_content_grounded_on_content_chain = create_is_distilled_content_grounded_on_content_chain()

def is_distilled_content_grounded_on_content(state):
    print("--------------------")

    """
    Determines if the distilled content is grounded on the original context.

    Args:
        distilled_content: The distilled content.
        original_context: The original context.

    Returns:
        Whether the distilled content is grounded on the original context.
    """

    print("Determining if the distilled content is grounded on the original context...")
    distilled_content = state["relevant_context"]
    print("Distilled content:", distilled_content)
    original_context = state["context"]
    print("Original context:", original_context)

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context
    }

    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output.grounded

    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"
    
def retrieve_chunks_context_per_question(state):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    try:
        logger.debug("retrieve_chunks_context_per_question called")
        retrievers = get_retrievers()
        chunks_retriever = retrievers['chunks']
        chapters_retriever = retrievers['summaries']
        book_quotes_retriever = retrievers['quotes']
        # Retrieve relevant documents
        logger.info("Retrieving relevant chunks...")
        print("Retrieving relevant chunks...")
        question = state["question"]
        logger.debug(f"Question: {question[:50]}")

        docs = chunks_retriever.invoke(question)
        logger.debug(f"Retrieved {len(docs)} documents")

        # Concatenate document content
        context = " ".join(doc.page_content for doc in docs)
        logger.info(f"Concatenated context length: {len(context)} characters")
        # context = escape_quotes(context)
        return {"context": context, "question": question}
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in retrieve_chunks_context_per_question: {str(e)}", exc_info=True)
        raise


def retrieve_summaries_context_per_question(state):
    retrievers = get_retrievers()
    chunks_retriever = retrievers['chunks']
    chapters_retriever = retrievers['summaries']
    book_quotes_retriever = retrievers['quotes']

    print("Retrieving relevant chapter summaries...")
    question = state["question"]

    docs_summaries = chapters_retriever.invoke(state["question"])

    # Concatenate chapter summaries with citation information
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )
    # context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}

def retrieve_book_quotes_context_per_question(state):
    question = state["question"]
    retrievers = get_retrievers()
    chunks_retriever = retrievers['chunks']
    chapters_retriever = retrievers['summaries']
    book_quotes_retriever = retrievers['quotes']

    print("Retrieving relevant book quotes...")
    docs_book_quotes = book_quotes_retriever.invoke(state["question"])
    book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)
    # book_qoutes_context = escape_quotes(book_qoutes)

    return {"context": book_qoutes, "question": question}
class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    context: str
    relevant_context: str


def create_qualitative_retrieval_book_chunks_workflow_app():
    qualitative_chunks_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

    # Define the nodes
    qualitative_chunks_retrieval_workflow.add_node("retrieve_chunks_context_per_question",retrieve_chunks_context_per_question)
    qualitative_chunks_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

    # Build the graph
    qualitative_chunks_retrieval_workflow.set_entry_point("retrieve_chunks_context_per_question")

    qualitative_chunks_retrieval_workflow.add_edge("retrieve_chunks_context_per_question", "keep_only_relevant_content")

    qualitative_chunks_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"grounded on the original context":END,
        "not grounded on the original context":"keep_only_relevant_content"},
        )

    
    qualitative_chunks_retrieval_workflow_app = qualitative_chunks_retrieval_workflow.compile()
    return qualitative_chunks_retrieval_workflow_app


def create_qualitative_retrieval_chapter_summaries_workflow_app():
    qualitative_summaries_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

    # Define the nodes
    qualitative_summaries_retrieval_workflow.add_node("retrieve_summaries_context_per_question",retrieve_summaries_context_per_question)
    qualitative_summaries_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

    # Build the graph
    qualitative_summaries_retrieval_workflow.set_entry_point("retrieve_summaries_context_per_question")

    qualitative_summaries_retrieval_workflow.add_edge("retrieve_summaries_context_per_question", "keep_only_relevant_content")

    qualitative_summaries_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"grounded on the original context":END,
        "not grounded on the original context":"keep_only_relevant_content"},
        )


    qualitative_summaries_retrieval_workflow_app = qualitative_summaries_retrieval_workflow.compile()
    return qualitative_summaries_retrieval_workflow_app


def create_qualitative_book_quotes_retrieval_workflow_app():
    qualitative_book_quotes_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)

    # Define the nodes
    qualitative_book_quotes_retrieval_workflow.add_node("retrieve_book_quotes_context_per_question",retrieve_book_quotes_context_per_question)
    qualitative_book_quotes_retrieval_workflow.add_node("keep_only_relevant_content",keep_only_relevant_content)

    # Build the graph
    qualitative_book_quotes_retrieval_workflow.set_entry_point("retrieve_book_quotes_context_per_question")

    qualitative_book_quotes_retrieval_workflow.add_edge("retrieve_book_quotes_context_per_question", "keep_only_relevant_content")

    qualitative_book_quotes_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"grounded on the original context":END,
        "not grounded on the original context":"keep_only_relevant_content"},
        )

    qualitative_book_quotes_retrieval_workflow_app = qualitative_book_quotes_retrieval_workflow.compile()

    return qualitative_book_quotes_retrieval_workflow_app



is_grounded_on_facts_chain = create_is_grounded_on_facts_chain()

def is_answer_grounded_on_context(state):
    """Determines if the answer to the question is grounded in the facts.
    
    Args:
        state: A dictionary containing the context and answer.
    """
    print("Checking if the answer is grounded in the facts...")
    context = state["context"]
    answer = state["answer"]
    
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"


def create_qualitative_answer_workflow_app():
    class QualitativeAnswerGraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        question: str
        context: str
        answer: str

    qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

    # Define the nodes

    qualitative_answer_workflow.add_node("answer_question_from_context",generate_answer_from_context)

    # Build the graph
    qualitative_answer_workflow.set_entry_point("answer_question_from_context")

    qualitative_answer_workflow.add_conditional_edges(
    "answer_question_from_context",is_answer_grounded_on_context ,{"hallucination":"answer_question_from_context", "grounded on context":END}

    )

    qualitative_answer_workflow_app = qualitative_answer_workflow.compile()
    return qualitative_answer_workflow_app


import mlflow
class PlanExecute(TypedDict):
    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    aggregated_context: str
    tool: str
    response: str
    embedding_cache: 'EmbeddingCache'
    # Phase 2 fields
    confidence: float
    is_grounded: bool
    replan_count: int

class Plan(BaseModel):
        """Plan to follow in future"""

        steps: List[str] = Field(
            description="different steps to follow, should be in sorted order"
        )

def create_plan_chain():
    

    planner_prompt =""" For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    """
    planner_prompt = PromptTemplate(
        template=planner_prompt,
        input_variables=["question"], 
        )

    planner_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)

    planner = planner_prompt | planner_llm.with_structured_output(Plan, method="json_schema")
    return planner


def create_break_down_plan_chain():

    break_down_plan_prompt_template = """You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
    you need to go through the plan refine it according to this:
    1. every step has to be able to be executed by either:
        i. retrieving relevant information from a vector store of book chunks
        ii. retrieving relevant information from a vector store of chapter summaries
        iii. retrieving relevant information from a vector store of book quotes
        iv. answering a question from a given context.
    2. every step should contain all the information needed to execute it.

    output the refined plan
    """

    break_down_plan_prompt = PromptTemplate(
        template=break_down_plan_prompt_template,
        input_variables=["plan"],
    )

    break_down_plan_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.0)

    break_down_plan_chain = break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan, method="json_schema")

    return break_down_plan_chain

def create_replanner_chain():
    # class ActPossibleResults(BaseModel):
    #     """Possible results of the action."""
    #     plan: Plan = Field(description="Plan to follow in future.")
    #     explanation: str = Field(description="Explanation of the action.")
        

    # act_possible_results_parser = JsonOutputParser(pydantic_object=ActPossibleResults)

    replanner_prompt_template =""" For the given objective, come up with a simple step by step plan of how to figure out the answer. 
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

    Your objective was this:
    {question}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    You already have the following context:
    {aggregated_context}

    Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
    Do not return previously done steps as part of the plan.

    the format is json so escape quotes and new lines.

    """

    replanner_prompt = PromptTemplate(
        template=replanner_prompt_template,
        input_variables=["question", "plan", "past_steps", "aggregated_context"],
        # partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
    )

    replanner_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.0)


    replanner = replanner_prompt | replanner_llm.with_structured_output(Plan, method="json_schema")
    return replanner

def create_task_handler_chain():
    tasks_handler_prompt_template = """You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
    You have the following tools at your disposal:
    1. retrieve_chunks: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
    - use retrieve_chunks when you think the current task should search for information in the book chunks.
    2. retrieve_summaries: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
    - use retrieve_summaries when you think the current task should search for information in the chapter summaries.
    3. retrieve_quotes: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
    - use retrieve_quotes when you think the current task should search for information in the book quotes.
    4. answer_from_context: a tool that answers a question from a given context.
    - use answer_from_context ONLY when you the current task can be answered by the aggregated context {aggregated_context}

    you also receive the last tool used {last_tool}
    if {last_tool} was retrieve_chunks, use other tools than retrieve_chunks.

    You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
    You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
    if you decide to use Tools retrieve_chunks,retrieve_summaries or retrieve_quotes, output the query to be used for the tool and also output the relevant tool.
    if you decide to use Tool answer_from_context, output the question to be used for the tool, the context, and also that the tool to be used is Tool answer_from_context.

    """

    class TaskHandlerOutput(BaseModel):
        """Output schema for the task handler."""
        query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
        curr_context: str = Field(description="The context to be based on in order to answer the query.")
        tool: str = Field(description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context.")


    task_handler_prompt = PromptTemplate(
        template=tasks_handler_prompt_template,
        input_variables=["curr_task", "aggregated_context", "last_tool", "past_steps", "question"],
    )

    task_handler_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.0)
    task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput, method="json_schema")
    return task_handler_chain

def create_anonymize_question_chain():
    class AnonymizeQuestion(BaseModel):
        """Anonymized question and mapping."""
        anonymized_question: str = Field(description="Anonymized question.")
        mapping: Dict[str, str] = Field(description="Mapping of original name entities to variables.")
        explanation: str = Field(description="Explanation of the action.")

    anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)


    anonymize_question_prompt_template = """ You are a question anonymizer. The input You receive is a string containing several words that construct a question {question}. Your goal is to changes all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
    ```example1:
            if the input is \"who is harry potter?\" the output should be \"who is X?\" and the mapping should be {{\"X\": \"harry potter\"}} ```
    ```example2:
            if the input is \"how did the bad guy played with the alex and rony?\"
            the output should be \"how did the X played with the Y and Z?\" and the mapping should be {{\"X\": \"bad guy\", \"Y\": \"alex\", \"Z\": \"rony\"}}```
    you must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
    output the anonymized question and the mapping as two separate fields in a json format as described here, without any additional text apart from the json format.
   """


    anonymize_question_prompt = PromptTemplate(
        template=anonymize_question_prompt_template,
        input_variables=["question"],
        partial_variables={"format_instructions": anonymize_question_parser.get_format_instructions()},
    )

    anonymize_question_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.0)
    anonymize_question_chain = anonymize_question_prompt | anonymize_question_llm | anonymize_question_parser
    return anonymize_question_chain


def create_deanonymize_plan_chain():
    class DeAnonymizePlan(BaseModel):
        """Possible results of the action."""
        plan: List[str] = Field(description="Plan to follow in future. with all the variables replaced with the mapped words.")

    de_anonymize_plan_prompt_template = """ you receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. you also receive
    the mapping for those variables to words {mapping}. replace all the variables in the list of tasks with the mapped words. if no variables are present,
    return the original list of tasks. in any case, just output the updated list of tasks in a json format as described here, without any additional text apart from the
    """


    de_anonymize_plan_prompt = PromptTemplate(
        template=de_anonymize_plan_prompt_template,
        input_variables=["plan", "mapping"],
    )
    

    de_anonymize_plan_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.7)
    de_anonymize_plan_chain = de_anonymize_plan_prompt | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan, method="json_schema")
    return de_anonymize_plan_chain

def create_can_be_answered_already_chain():
    class CanBeAnsweredAlready(BaseModel):
        """Possible results of the action."""
        can_be_answered: bool = Field(description="Whether the question can be fully answered or not based on the given context.")

    can_be_answered_already_prompt_template = """You receive a query: {question} and a context: {context}.
    You need to determine if the question can be fully answered relying only the given context.
    The only infomation you have and can rely on is the context you received. 
    you have no prior knowledge of the question or the context.
    if you think the question can be answered based on the context, output 'true', otherwise output 'false'.
    """

    can_be_answered_already_prompt = PromptTemplate(
        template=can_be_answered_already_prompt_template,
        input_variables=["question","context"],
    )

    can_be_answered_already_llm = get_llm(model="minimaxai/minimax-m3", temperature=0.0)
    can_be_answered_already_chain = can_be_answered_already_prompt | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready, method="json_schema")
    return can_be_answered_already_chain


task_handler_chain = create_task_handler_chain()
qualitative_chunks_retrieval_workflow_app = create_qualitative_retrieval_book_chunks_workflow_app()
qualitative_summaries_retrieval_workflow_app = create_qualitative_retrieval_chapter_summaries_workflow_app()
qualitative_book_quotes_retrieval_workflow_app = create_qualitative_book_quotes_retrieval_workflow_app()
qualitative_answer_workflow_app = create_qualitative_answer_workflow_app()
de_anonymize_plan_chain = create_deanonymize_plan_chain()
planner = create_plan_chain()
break_down_plan_chain = create_break_down_plan_chain()
replanner = create_replanner_chain()
anonymize_question_chain = create_anonymize_question_chain()
can_be_answered_already_chain = create_can_be_answered_already_chain()

@mlflow.trace(span_type="task_handler_chain")
def run_task_handler_chain(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    try:
        logger.debug("run_task_handler_chain called")
        state["curr_state"] = "task_handler"
        logger.info("Current plan:")
        logger.debug(f"Plan: {state['plan']}")
        # print("--------------------")

        if "past_steps" not in state or state['past_steps'] is None:
            state["past_steps"] = []
            logger.debug("Initialized past_steps")

        curr_task = state["plan"][0]
        logger.debug(f"Current task: {curr_task}")

        inputs = {"curr_task": curr_task,
                   "aggregated_context": state.get("aggregated_context", ""),
                   "last_tool": state.get("tool", ""),
                   "past_steps": state["past_steps"],
                   "question": state["question"]}

        logger.info("Invoking task handler chain...")
        output = task_handler_chain.invoke(inputs)
        logger.debug(f"Task handler output tool: {output.tool}")

        state["past_steps"].append(curr_task)
        state["plan"].pop(0)
        logger.debug(f"Remaining plan items: {len(state['plan'])}")

        if output.tool == "retrieve_chunks":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"]="retrieve_chunks"
            logger.info("Selected retrieve_chunks tool")

        elif output.tool == "retrieve_summaries":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"]="retrieve_summaries"
            logger.info("Selected retrieve_summaries tool")

        elif output.tool == "retrieve_quotes":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"]="retrieve_quotes"
            logger.info("Selected retrieve_quotes tool")

        elif output.tool == "answer_from_context":
            state["query_to_retrieve_or_answer"] = output.query
            state["curr_context"] = output.curr_context
            state["tool"]="answer"
            logger.info("Selected answer_from_context tool")
        else:
            logger.error(f"Invalid tool selected: {output.tool}")
            raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")

        logger.info("Task handler chain execution completed successfully")
        return state
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in run_task_handler_chain: {str(e)}", exc_info=True)
        raise  


@mlflow.trace(span_type="retrieve_or_answer")
def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    state["curr_state"] = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state["tool"] == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")  


@mlflow.trace(span_type="qualitative_chunks_retrieval_workflow")
def run_qualitative_chunks_retrieval_workflow(state):
    """
    Run the qualitative chunks retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    try:
        logger.debug("run_qualitative_chunks_retrieval_workflow called")
        state["curr_state"] = "retrieve_chunks"
        logger.info("Running the qualitative chunks retrieval workflow...")
        print("Running the qualitative chunks retrieval workflow...")

        question = state["query_to_retrieve_or_answer"]
        logger.debug(f"Query: {question[:50]}")

        inputs = {"question": question}
        logger.debug("Streaming workflow...")
        relevant_content = ""
        for output in qualitative_chunks_retrieval_workflow_app.stream(inputs):
            node_output = list(output.values())[0] if output else {}
            if isinstance(node_output, dict) and "relevant_context" in node_output:
                relevant_content = node_output["relevant_context"]
            print("--------------------")

        if "aggregated_context" not in state or state["aggregated_context"] is None:
            state["aggregated_context"] = ""
            logger.debug("Initialized aggregated_context")

        if relevant_content:
            state["aggregated_context"] += relevant_content
            logger.info(f"Updated aggregated_context to {len(state['aggregated_context'])} characters")
        return state
    except Exception as e:
        logger.error(f"Error in run_qualitative_chunks_retrieval_workflow: {str(e)}", exc_info=True)
        raise

@mlflow.trace(span_type="qualitative_summaries_retrieval_workflow")
def run_qualitative_summaries_retrieval_workflow(state):
    """
    Run the qualitative summaries retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_summaries"
    print("Running the qualitative summaries retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    relevant_content = ""
    for output in qualitative_summaries_retrieval_workflow_app.stream(inputs):
        node_output = list(output.values())[0] if output else {}
        if isinstance(node_output, dict) and "relevant_context" in node_output:
            relevant_content = node_output["relevant_context"]
        print("--------------------")
    if "aggregated_context" not in state or state["aggregated_context"] is None:
        state["aggregated_context"] = ""
    if relevant_content:
        state["aggregated_context"] += relevant_content
    return state


@mlflow.trace(span_type="qualitative_book_quotes_retrieval_workflow")
def run_qualitative_book_quotes_retrieval_workflow(state):
    """
    Run the qualitative book quotes retrieval workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_book_quotes"
    print("Running the qualitative book quotes retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    relevant_content = ""
    for output in qualitative_book_quotes_retrieval_workflow_app.stream(inputs):
        node_output = list(output.values())[0] if output else {}
        if isinstance(node_output, dict) and "relevant_context" in node_output:
            relevant_content = node_output["relevant_context"]
        print("--------------------")
    if "aggregated_context" not in state or state["aggregated_context"] is None:
        state["aggregated_context"] = ""
    if relevant_content:
        state["aggregated_context"] += relevant_content
    return state
   

@mlflow.trace(span_type="qualitative_answer_workflow")
def run_qualtative_answer_workflow(state):
    """
    Run the qualitative answer workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "answer"
    print("Running the qualitative answer workflow...")
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass 
        print("--------------------")
    if "aggregated_context" not in state or state["aggregated_context"] is None:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["answer"]
    return state


@mlflow.trace(span_type="qualitative_answer_workflow_for_final_answer")
def run_qualtative_answer_workflow_for_final_answer(state):
    """
    Run the qualitative answer workflow for the final answer.
    Args:
        state: The current state of the plan execution.
    Returns:
        The state with the updated response.
    """
    state["curr_state"] = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, value in output.items():
            pass  
        print("--------------------")
    state["response"] = value
    return state

@mlflow.trace(span_type="anonymize_queries")
def anonymize_queries(state: PlanExecute):
    """
    Anonymizes the question.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the anonymized question and mapping.
    """
    try:
        logger.debug("anonymize_queries called")
        state["curr_state"] = "anonymize_question"
        logger.info(f"Anonymizing question: {state['question'][:50]}")
        print("state['question']: ", state['question'])
        print("Anonymizing question")
        print("--------------------")

        input_values = {"question": state['question']}
        logger.debug("Invoking anonymize_question_chain...")
        anonymized_question_output = anonymize_question_chain.invoke(input_values)
        logger.debug(f"Chain output: {anonymized_question_output}")
        print(f'anonymized_question_output: {anonymized_question_output}')

        anonymized_question = anonymized_question_output["anonymized_question"]
        logger.info(f"Anonymized question: {anonymized_question}")
        print(f'anonimized_querry: {anonymized_question}')
        print("--------------------")

        mapping = anonymized_question_output["mapping"]
        logger.debug(f"Mapping: {mapping}")
        state["anonymized_question"] = anonymized_question
        state["mapping"] = mapping

        logger.info("Anonymization completed successfully")
        return state
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in anonymize_queries: {str(e)}", exc_info=True)
        raise

@mlflow.trace(span_type="deanonymize_queries")
def deanonymize_queries(state: PlanExecute):
    """
    De-anonymizes the plan.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the de-anonymized plan.
    """
    state["curr_state"] = "de_anonymize_plan"
    print("De-anonymizing plan")
    print("--------------------")
    deanonimzed_plan = de_anonymize_plan_chain.invoke({"plan": state["plan"], "mapping": state["mapping"]})
    state["plan"] = deanonimzed_plan.plan
    print(f'de-anonimized_plan: {deanonimzed_plan.plan}')
    return state


@mlflow.trace(span_type="plan_step")
def plan_step(state: PlanExecute):
    """
    Plans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    try:
        logger.debug("plan_step called")
        state["curr_state"] = "planner"
        logger.info("Planning step")
        print("Planning step")
        print("--------------------")

        logger.debug(f"Anonymized question: {state['anonymized_question'][:50]}")
        plan = planner.invoke({"question": state['anonymized_question']})
        logger.debug("Planner chain invoked successfully")

        state["plan"] = plan.steps
        logger.info(f"Generated plan with {len(state['plan'])} steps")
        print(f'plan: {state["plan"]}')
        logger.debug(f"Plan steps: {state['plan']}")

        return state
    except KeyError as e:
        logger.error(f"Missing key in state: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in plan_step: {str(e)}", exc_info=True)
        raise

@mlflow.trace(span_type="break_down_plan_step")
def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    print("--------------------")
    refined_plan = break_down_plan_chain.invoke(state["plan"])
    state["plan"] = refined_plan.steps
    return state


def should_replan(state: PlanExecute):
    """Conditional routing: determine if replanning is needed based on confidence."""
    confidence = state.get('confidence', 0.0)
    is_grounded = state.get('is_grounded', False)
    replan_count = state.get('replan_count', 0)

    if replan_count >= 2:
        logger.warning(f"Max replan iterations (2) reached, forcing exit")
        return "can_be_answered"

    if confidence > 0.85 and is_grounded:
        logger.info(f"High confidence answer ({confidence:.2f}), skipping replan")
        return "can_be_answered"
    else:
        logger.info(f"Low confidence ({confidence:.2f}), will replan")
        return "replan"


def replan_step(state: PlanExecute):
    """
    Replans the next step.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "replan"
    print("Replanning step")
    print("--------------------")
    inputs = {"question": state["question"], "plan": state["plan"], "past_steps": state.get("past_steps", []), "aggregated_context": state["aggregated_context"]}
    plan = replanner.invoke(inputs)
    state["plan"] = plan.steps
    return state


def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.
    Args:
        state: The current state of the plan execution.
    Returns:
        whether the original question can be answered or not.
    """
    state["curr_state"] = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    print("--------------------")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    output = can_be_answered_already_chain.invoke(inputs)
    if output.can_be_answered == True:
        print("The ORIGINAL QUESTION can be fully answered already.")
        print("--------------------")
        print("the aggregated context is:")
        # print(text_wrap(state["aggregated_context"]))
        print(state["aggregated_context"])
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        print("--------------------")
        return "cannot_be_answered_yet"


@mlflow.trace(name="create_agent")
def create_agent():
    """Create and compile the agent workflow graph."""
    try:
        logger.info("Starting create_agent")
        agent_workflow = StateGraph(PlanExecute)
        logger.debug("Created StateGraph")

        # Initialize Phase 1 singleton caches (first call only, then reused)
        logger.info("Initializing Phase 1 caches...")
        _ = get_embeddings()  # Initialize embeddings
        _ = get_vector_stores()  # Initialize vector stores
        _ = get_retrievers()  # Initialize retrievers
        logger.info("Phase 1 caches initialized")

        # Add the anonymize node
        agent_workflow.add_node("anonymize_question", anonymize_queries)
        logger.debug("Added anonymize_question node")

        # Add the plan node
        agent_workflow.add_node("planner", plan_step)
        logger.debug("Added planner node")

        # Add the break down plan node
        agent_workflow.add_node("break_down_plan", break_down_plan_step)
        logger.debug("Added break_down_plan node")

        # Add the deanonymize node
        agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)
        logger.debug("Added de_anonymize_plan node")

        # Add the qualitative chunks retrieval node
        agent_workflow.add_node("retrieve_chunks", run_qualitative_chunks_retrieval_workflow)
        logger.debug("Added retrieve_chunks node")

        # Add the qualitative summaries retrieval node
        agent_workflow.add_node("retrieve_summaries", run_qualitative_summaries_retrieval_workflow)
        logger.debug("Added retrieve_summaries node")

        # Add the qualitative book quotes retrieval node
        agent_workflow.add_node("retrieve_book_quotes", run_qualitative_book_quotes_retrieval_workflow)
        logger.debug("Added retrieve_book_quotes node")

        # Add the qualitative answer node
        agent_workflow.add_node("answer", run_qualtative_answer_workflow)
        logger.debug("Added answer node")

        # Add the task handler node
        agent_workflow.add_node("task_handler", run_task_handler_chain)
        logger.debug("Added task_handler node")

        # Add a replan node
        agent_workflow.add_node("replan", replan_step)
        logger.debug("Added replan node")

        # Add answer from context node
        agent_workflow.add_node("get_final_answer", run_qualtative_answer_workflow_for_final_answer)
        logger.debug("Added get_final_answer node")

        # Set the entry point
        agent_workflow.set_entry_point("anonymize_question")
        logger.debug("Set entry point to anonymize_question")

        # From anonymize we go to plan
        agent_workflow.add_edge("anonymize_question", "planner")

        # From plan we go to deanonymize
        agent_workflow.add_edge("planner", "de_anonymize_plan")

        # From deanonymize we go to break down plan
        agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")

        # From break_down_plan we go to task handler
        agent_workflow.add_edge("break_down_plan", "task_handler")

        # From task handler we go to either retrieve or answer
        agent_workflow.add_conditional_edges("task_handler", retrieve_or_answer, {"chosen_tool_is_retrieve_chunks": "retrieve_chunks", "chosen_tool_is_retrieve_summaries":
                                                                                "retrieve_summaries", "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes", "chosen_tool_is_answer": "answer"})
        logger.debug("Added conditional edges from task_handler")

        # After retrieving we go to replan
        agent_workflow.add_edge("retrieve_chunks", "replan")
        agent_workflow.add_edge("retrieve_summaries", "replan")
        agent_workflow.add_edge("retrieve_book_quotes", "replan")

        # After answering we go to replan (original sequential flow - Phase 1 keeps flow unchanged)
        agent_workflow.add_edge("answer", "replan")
        logger.debug("Added edge from answer to replan")

        # After replanning we check if the question can be answered
        agent_workflow.add_conditional_edges("replan", can_be_answered, {"can_be_answered_already": "get_final_answer", "cannot_be_answered_yet": "break_down_plan"})
        logger.debug("Added conditional edges from replan")

        # After getting the final answer we end
        agent_workflow.add_edge("get_final_answer", END)

        logger.info("Compiling agent workflow...")
        plan_and_execute_app = agent_workflow.compile()
        logger.info("Agent workflow compiled successfully")

        return plan_and_execute_app
    except Exception as e:
        logger.error(f"Error in create_agent: {str(e)}", exc_info=True)
        raise

