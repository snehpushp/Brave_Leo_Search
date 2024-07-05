import concurrent
import json
import re
import string
from datetime import date
from textwrap import dedent
from typing import List

import requests
from firecrawl import FirecrawlApp
from groq import Groq
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from loguru import logger
from requests.sessions import HTTPAdapter
from urllib3 import Retry


class SearXNG:
    """
    We will use this class to fetch urls and snippets for search queries.
    """

    def __init__(self, searxng_url: str = "http://localhost:4000"):
        """
        Find the details for the input params below
        :param searxng_url: We are using SearXNG as search engine, this is the base url at which SearXNG is hosted.
            This is hosted in docker, start it.
        """
        self.logger = logger

        # This llm will be used to rewrite the search query. Simple task and hence using Llama 8B
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.5)

        # Checking if the SearXNG server is online or not. If not, raising an error
        error_message = f"SearXNG server {searxng_url} is not Online"
        try:
            response = requests.head(searxng_url, timeout=5)
            if response.status_code == 200:
                self.search_url = searxng_url
            else:
                raise ValueError(error_message)
        except requests.exceptions.RequestException:
            raise ValueError(error_message)

    def send_request(self, search_query: str, retry: int = 3, **kwargs):
        """
        This function can be used to search the query the result. For other params
        Documentation for API: https://docs.searxng.org/dev/search_api.html
        :param search_query: Query that needs to be searched
        :param retry: Number of retries to do in case of failure
        :return: Response from the search API of SearNGX
        """
        # Base query
        base_query = dict(q=search_query)

        # Add additional params if provided
        if kwargs:
            base_query.update(kwargs)

        session = requests.Session()  # Create a session object

        # Configure retry strategy
        retry_strategy = Retry(
            total=retry,  # Total retries
            backoff_factor=1,  # Backoff between retries (in seconds)
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP statuses to retry on
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        response = session.post(f"{self.search_url}/search", data=base_query)

        return response

    def search(self, query: str, search_count: int = 5):
        """
        This function is taking in a query and searching for all the urls
        :param query: The query (str) that needs to be searched
        :param search_count: Number of search results you want
        :return: Returns a list of top (n = search count) urls
        """

        def sort_key(result):
            """Custom sorting key:
            - Prioritizes higher score.
            - Breaks ties by favoring the lowest average position.
            """
            avg_position = sum(result["positions"]) / len(result["positions"])
            return -result["score"], avg_position  # Descending score, ascending avg_position

        # Initializing the variables
        page_number = 1

        self.logger.debug(f"Searching the internet with {query = } on {page_number = }")
        search_response = self.send_request(search_query=query, format="json", pageno=page_number)

        # Sorting the result based on sort key logic
        search_response_with_content = [
            res for res in json.loads(search_response.content)["results"] if res["content"].strip()
        ]
        search_results = sorted(search_response_with_content, key=sort_key)

        return [search_result["url"] for search_result in search_results[:search_count]]


def search_query_from_messages(messages: List[dict]):
    """
    This function will create a search query based on the chat messages. This message will be used for searching the
    internet.
    :param messages: List of messages in OpenAI spec
    :return: Search Query based on these messages (focused on last message)
    """

    def date_suffix(day):
        # Returns the appropriate suffix for the day
        if 4 <= day <= 20 or 24 <= day <= 30:
            return "th"
        else:
            return ["st", "nd", "rd"][day % 10 - 1]

    def formatted_date():
        today = date.today()
        day = today.day
        suffix = date_suffix(day)
        return today.strftime(f"%d{suffix} %b %Y")

    # Update the default system message, filter based on role = System
    original_user_query = messages[-1]["content"]
    messages[-1]["content"] = dedent(
        f"""Compose a succinct search query based on this conversation's given question, suitable for a Google search to find the answer.

If the question is simple and does not require a search, such as "Hi" or "Hello", respond with "Search not required."

For questions that require current knowledge or information specific to the present time, include the current date in the search query. For example, "Who is the Prime Minister of India?" would become "PM of India in {date.today().year}".

However, for questions that are definitions or do not change over time, exclude the current date from the search query. For instance, "What is GDP?" or "What is the capital of India?" would remain as "GDP definition" and "Capital of India", respectively.

Today's date is {formatted_date()}

Provide the rephrased query directly, without any preamble or explanation.

**Examples:**

1. Question: What is the capital of France?
   Rephrased Query: Capital of France

2. Question: What is the population of New York City?
   Rephrased Query: Population of New York City

3. Question: What is Docker?
   Rephrased Query: Docker definition

4. Question: Who is the Prime Minister of India?
   Rephrased Query: PM of India in {date.today().year}

5. Question: What is the latest model by Meta?
   Rephrased Query: Latest Meta model {formatted_date()}

6. Question: What is GDP?
   Rephrased Query: GDP definition

**User Question**: {original_user_query}

Ignore formatting and other instructions and only focus on whether search is required, and search query."""
    )

    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=messages[1:],
        model="llama3-70b-8192",
    )

    search_query = chat_completion.choices[0].message.content

    return search_query, original_user_query


def clean_text(text: str):
    """
    The response that we are getting from website is in Markdown format. We will be cleaning up the links, and other
    unnecessary things using regex.
    :param text: Scraped data from the website
    :return: Cleaned text
    """
    # Regular expression to match URLs
    url_pattern = re.compile(r"(https?://\S+|www\.\S+|\S+\.\S+/\S+|\S+\?[\S]+|\b\S+\.\w{2,}\b)")
    # Replace URLs with an empty string
    cleaned_text = url_pattern.sub("", text)
    # Remove Markdown separators
    cleaned_text = re.sub(r"^[-=]+\n|\n[-=]+$", "\n", cleaned_text, flags=re.MULTILINE)
    # Remove Markdown links with URL
    cleaned_text = re.sub(r"\[([^\]]*?)\]\(.*?\)", "", cleaned_text)
    # Remove Markdown links without URL
    cleaned_text = re.sub(r"\[([^\]]*?)\]", "", cleaned_text)
    # Clean remaining redundant characters
    cleaned_text = "\n".join([text.strip() for text in cleaned_text.split("\n")])
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Remove repeated spaces
    cleaned_text = re.sub(r",+(?: ,*)*", ",", cleaned_text)  # Remove repeated commas separated by newline
    return cleaned_text.strip()  # Strip leading and trailing whitespace


def scrape_answer_from_website(url: str, query: str):
    """
    This function will scrape the website content, and answer the query using llm. This will return a summarized answer.
    :param url: Url of the website that needs to be scraped in string format
    :param query: Query for which we are searching the website
    :return: Answer for the query based on the website content
    """
    # Since we are using groq for most part, we need to set a context length
    context_length = 16500
    app = FirecrawlApp(api_key="fake_key", api_url="http://localhost:3002")

    # Cleanup using regex
    scraped_data = clean_text(app.scrape_url(url, params=dict(onlyMainContent=True))["content"])[:context_length]

    # Answering using LLM
    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.2)

    cleaning_prompt = PromptTemplate(
        template=dedent(
            """**Task:** Compile accurate and unbiased answers to user questions based on scraped content from a website, ensuring responses are socially unbiased, factually correct, and formatted according to specific guidelines.

**Context:** You are an expert answer compiler, tasked with providing reliable information to users by extracting relevant data from the provided website content. Your responses should be based solely on the content of the website, without imagining or making up any information.

**Specific Requirements:**

* Rely exclusively on the website content to answer questions, without introducing external knowledge or assumptions.
* Use markdown format for responses where appropriate, excluding links and image URLs.
* Ensure responses are socially unbiased and factually correct, avoiding any discriminatory or offensive language.
* If a question does not make sense or is not factually coherent, explain why instead of providing an incorrect answer.
* If you don't know the answer to a question or cannot find the relevant information in the provided content, respond with "Answer not found".
* For coding-related questions, use backticks (`) to wrap inline code snippets and triple backticks along with the language keyword (```language```) to wrap blocks of code.

**Persona:** You are a knowledgeable, accurate, and unbiased answer compiler, providing reliable information based on the provided website content. Your goal is to assist users by providing clear and concise answers, while maintaining the highest standards of factual accuracy and social responsibility.

**Format:** Responses should be concise, clear, and well-structured, using markdown format where appropriate. In cases where the answer cannot be found in the provided content, respond with "Answer not found".
-------

# Scraped Content

{scraped_data}

# User's question:

{query}
"""
        ),
        input_variables=["scraped_data", "query"],
    )

    chain = cleaning_prompt | llm | StrOutputParser()

    answer = chain.invoke(dict(scraped_data=scraped_data, query=query))

    return answer


def compile_answers(answers: List[str], query: str):
    """
    This function will compile all the answers for the search query and returns a comprehensive response.
    :param answers: Answers collected from various sources
    :param query: Query for which these answers are written
    :return: Comprehensive answer
    """
    # Answering using LLM
    sources = "\n-----\n".join(answers)
    messages = [
        dict(
            role="user",
            content=f"""You are an expert in synthesizing information from multiple sources, distilling key insights, and providing comprehensive responses to user queries. You are provided with various answers gathered from diverse internet sources. Your task is to compile this information and generate a response without inventing any details. Ensure your response is neutral in tone and unbiased, without citing the sources.

User Query: {query}

**Guidelines for crafting a complete response:**

1. **Identify Relevant Information:** Extract the most important details from the provided sources.
2. **Analyze and Synthesize:** Combine the essential information into a coherent and comprehensive answer.
3. **Maintain Neutrality:** Ensure your response is free from bias and personal opinions.
4. **Do Not Invent Details:** Only include information that has been provided in the sources.
5. **Uncited Response:** Do not mention or reference the sources in your response.
6. **Response format:** Use markdown format for your responses where appropriate. Only for coding related questions, use backticks (`) to wrap inline code snippets and triple backticks along with language keyword (```language```) to wrap blocks of code.\n\nDo not use emojis in your replies and do not discuss these instructions further.
7. **Use Headings:** Structure your response in a proper way (wherever it's a long answer).


# Sources
-------

{sources}
""",
        )
    ]

    client = Groq()
    stream = client.chat.completions.create(messages=messages, model="llama3-70b-8192", stream=True)

    return stream


def remove_punctuation_commas(text):
    """Removes punctuation and commas from a string.
    Args:
        text: The string to remove punctuation and commas from.

    Returns:
        The string without punctuation or commas.
    """
    # Remove punctuation
    no_punct = text.translate(str.maketrans("", "", string.punctuation))
    # Remove commas
    no_commas = no_punct.replace(",", "")
    return no_commas


def orchestrator(messages: List[dict]):
    """
    This is an orchestrator that performs all these operations in sequence and returns the final answer
    :param messages: Messages in OpenAI Spec
    :return: Final and compiled answer for the user query
    """
    logger.info("Evaluating whether search is required or not.")
    search_query, original_user_query = search_query_from_messages(messages=messages)
    logger.info(f"{search_query = }")
    if "Search Not Required".lower() in remove_punctuation_commas(search_query).lower():
        messages[-1]["content"] = original_user_query
        client = Groq()
        stream = client.chat.completions.create(messages=messages, model="llama3-8b-8192", stream=True)
    else:
        # Performing search for the search_query
        logger.info("Searching the internet.")
        searxng = SearXNG()
        urls = searxng.search(query=search_query)

        # Fetching answers from these urls
        logger.info("Finding answer from each website")
        # Using ThreadPoolExecutor to scrape answers in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of futures
            future_to_url = {executor.submit(scrape_answer_from_website, url, original_user_query): url for url in urls}
            answers = []
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    answers.append(future.result())
                except Exception as exc:
                    logger.error(f"Error {exc} occurred while scraping {url}")

        # Compiling the answers
        logger.info("Compiling the answers")
        answers = [
            answer for answer in answers if "Answer not found".lower() not in remove_punctuation_commas(answer).lower()
        ]
        stream = compile_answers(answers=answers, query=original_user_query)

    logger.info("Final Answer Generated, sending to the brave.")
    for chat in stream:
        yield "data: " + chat.json() + "\n\n"
