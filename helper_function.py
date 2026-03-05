import pypdf
from langchain_core.documents import Document
import re

# import pdfplumber

# with pdfplumber.open(r"c:\\Users\Anupam\\Desktop\\AI AGENTS\\SOPHISTICATED-AGENT\\Harry_Potter_Book_1_The_Sorcerers_Stone.pdf") as pdf:
#     total_chars = 0
#     non_empty_pages = 0
    
#     for i, page in enumerate(pdf.pages):
#         extracted = page.extract_text()
#         char_count = len(extracted) if extracted else 0
        
#         if char_count > 0:
#             non_empty_pages += 1
#             total_chars += char_count
#             print(f"Page {i+1}: {char_count} chars")
#             print(f"  Sample: {repr(extracted[:100])}")
#         else:
#             # Try extracting words as fallback
#             words = page.extract_words()
#             if words:
#                 word_text = " ".join([w["text"] for w in words])
#                 char_count = len(word_text)
#                 non_empty_pages += 1
#                 total_chars += char_count
#                 print(f"Page {i+1}: {char_count} chars (from words)")
#                 print(f"  Sample: {repr(word_text[:500])}")
#             else:
#                 print(f"Page {i+1}: EMPTY")
        
#         if i >= 8:
#             break
    
    # print(f"\nSummary: {non_empty_pages}/{min(6, len(pdf.pages))} pages had text")
def split_into_chapters(book_path):
    """
    Splits a PDF book into chapters based on chapter title patterns.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter number metadata.
    """
    with open(book_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        documents= pdf_reader.pages # Get all the pages in the PDF
        # Debug 1: Check page count
        print(f"Total pages: {len(documents)}")
        text = " ".join([doc.extract_text() for doc in documents]) # Extract text from all pages and concatenate
        # Split the text into chapters based on chapter title patterns (e.g., "Chapter 1", "Chapter 2", etc.)
        print(f"Extracted text length: {len(text)} characters") 
        print(f"First 500 chars: {text[:500]}")# Debug 2: Check extracted text length
        
        chapters = re.split(r'(Chapter\s+(?:[A-Z]+|\d+)(?:\s+[A-Z]+)*)', text, flags=re.IGNORECASE)
        chapter_num =1
        chapter_docs = []
        for i in range(1, len(chapters)-1, 2): # Start from 1 to skip any text before the first chapter, step by 2 to get chapter titles
            chapter_text = chapters[i] + chapters[i+1] # Combine chapter title and content
            doc = Document(page_content=chapter_text.strip(), metadata= {"chapter": chapter_num})
            chapter_docs.append(doc)
            chapter_num += 1
        return chapter_docs
    
    
def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def extract_book_quotes_as_documents(documents, min_length=50):
    quotes_as_documents = []
    # Correct pattern for quotes longer than min_length characters, including line breaks
    quote_pattern_longer_than_min_length = re.compile(rf'“(.{{{min_length},}}?)”', re.DOTALL)

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        found_quotes = quote_pattern_longer_than_min_length.findall(content)
        for quote in found_quotes:
            quote_doc = Document(page_content=quote)
            quotes_as_documents.append(quote_doc)
    
    return quotes_as_documents

def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text: The input text string.

    Returns:
        The text string with double newlines replaced by single newlines.
    """

    cleaned_text = re.sub(r'\n\n', '\n', text)  # Replace double newlines with single newlines
    return cleaned_text