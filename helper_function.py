import pypdf
from langchain_core.documents import Document
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('./logs/helper_function.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def split_into_chapters(book_path):
    """
    Splits a PDF book into chapters based on chapter title patterns.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter number metadata.
    """
    try:
        logger.debug(f"Starting split_into_chapters with book_path: {book_path}")
        with open(book_path, "rb") as file:
            logger.debug(f"Successfully opened file: {book_path}")
            pdf_reader = pypdf.PdfReader(file)
            documents= pdf_reader.pages # Get all the pages in the PDF
            # Debug 1: Check page count
            logger.info(f"Total pages: {len(documents)}")
            print(f"Total pages: {len(documents)}")
            text = " ".join([doc.extract_text() for doc in documents]) # Extract text from all pages and concatenate
            # Split the text into chapters based on chapter title patterns (e.g., "Chapter 1", "Chapter 2", etc.)
            logger.info(f"Extracted text length: {len(text)} characters")
            print(f"Extracted text length: {len(text)} characters")
            logger.debug(f"First 500 chars: {text[:500]}")
            print(f"First 500 chars: {text[:500]}")# Debug 2: Check extracted text length

            chapters = re.split(r'(Chapter\s+(?:[A-Z]+|\d+)(?:\s+[A-Z]+)*)', text, flags=re.IGNORECASE)
            logger.debug(f"Split text into {len(chapters)} sections")
            chapter_num =1
            chapter_docs = []
            for i in range(1, len(chapters)-1, 2): # Start from 1 to skip any text before the first chapter, step by 2 to get chapter titles
                chapter_text = chapters[i] + chapters[i+1] # Combine chapter title and content
                doc = Document(page_content=chapter_text.strip(), metadata= {"chapter": chapter_num})
                chapter_docs.append(doc)
                logger.debug(f"Created chapter document {chapter_num} with {len(chapter_text)} characters")
                chapter_num += 1
            logger.info(f"Successfully created {len(chapter_docs)} chapter documents")
            return chapter_docs
    except FileNotFoundError as e:
        logger.error(f"File not found: {book_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in split_into_chapters: {str(e)}", exc_info=True)
        raise
    
    
def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """
    try:
        logger.debug(f"Starting replace_t_with_space with {len(list_of_documents)} documents")
        for i, doc in enumerate(list_of_documents):
            original_length = len(doc.page_content)
            doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
            logger.debug(f"Document {i}: Replaced tabs in content (length: {original_length})")
        logger.info(f"Successfully replaced tabs in all {len(list_of_documents)} documents")
        return list_of_documents
    except AttributeError as e:
        logger.error(f"Document missing 'page_content' attribute: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in replace_t_with_space: {str(e)}", exc_info=True)
        raise

def extract_book_quotes_as_documents(documents, min_length=50):
    """Extract quoted text from documents as separate Document objects"""
    try:
        logger.debug(f"Starting extract_book_quotes_as_documents with {len(documents)} documents, min_length={min_length}")
        quotes_as_documents = []
        # Correct pattern for quotes longer than min_length characters, including line breaks
        quote_pattern_longer_than_min_length = re.compile(rf'”(.{{{min_length},}}?)”', re.DOTALL)
        logger.debug(f"Compiled quote pattern with minimum length {min_length}")

        for i, doc in enumerate(documents):
            try:
                content = doc.page_content
                content = content.replace('\n', ' ')
                found_quotes = quote_pattern_longer_than_min_length.findall(content)
                logger.debug(f"Document {i}: Found {len(found_quotes)} quotes")
                for j, quote in enumerate(found_quotes):
                    quote_doc = Document(page_content=quote)
                    quotes_as_documents.append(quote_doc)
                    logger.debug(f"Document {i}, Quote {j}: Extracted quote of length {len(quote)}")
            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}", exc_info=True)
                raise

        logger.info(f"Successfully extracted {len(quotes_as_documents)} quotes from {len(documents)} documents")
        return quotes_as_documents
    except Exception as e:
        logger.error(f"Error in extract_book_quotes_as_documents: {str(e)}", exc_info=True)
        raise

def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text: The input text string.

    Returns:
        The text string with double newlines replaced by single newlines.
    """
    try:
        logger.debug(f"Starting replace_double_lines_with_one_line with text length: {len(text)}")
        original_length = len(text)
        cleaned_text = re.sub(r'\n\n', '\n', text)  # Replace double newlines with single newlines
        new_length = len(cleaned_text)
        logger.info(f"Replaced double newlines: Original length {original_length}, New length {new_length}")
        logger.debug(f"Removed {original_length - new_length} characters")
        return cleaned_text
    except TypeError as e:
        logger.error(f"Text must be a string, got {type(text)}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in replace_double_lines_with_one_line: {str(e)}", exc_info=True)
        raise