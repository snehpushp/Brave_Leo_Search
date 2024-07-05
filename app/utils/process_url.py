import re
from textwrap import dedent

from firecrawl import FirecrawlApp
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq


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


def scrape_website_content(url: str):
    """
    Useful to scrape a website content, just pass a string with only the full url,
    no need for a final slash `/`, eg: https://google.com or https://clearbit.com/about-us
    :param url: Url of the website that needs to be scraped in string format
    :return: Scraped content of the website
    """
    # Since we are using groq for most part, we need to set a context length
    context_length = 16000  # This is in chars and not words.
    app = FirecrawlApp(api_key="fake_key", api_url="http://localhost:3002")

    scraped_data = clean_text(app.scrape_url(url, params=dict(onlyMainContent=True))["content"])

    # We will perform cleanup using regex and llm.
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)

    cleaning_prompt = PromptTemplate(
        template=dedent(
            """Clean a Markdown Article: Remove Non-Essential Elements

Take a Markdown article as input and remove all links, headers, footers, references, and other non-article elements while preserving the original content, structure, and formatting.
I don't want to lose any data, thus it's important to keep as much as we can. Refrain from removing content in middle of article.

Specifically, keep the following elements intact:

* Headings and subheadings
* Code blocks
* Tables
* JSON data

Do not summarize, paraphrase, or add any additional text, commentary, or explanations to the article.

Respond with the cleaned article content, starting from the title, without any preamble or introduction.
-------

# Scraped Content

{scraped_data}
"""
        ),
        input_variables=["scraped_data"],
    )

    chain = cleaning_prompt | llm | StrOutputParser()

    result = chain.invoke(dict(scraped_data=scraped_data))

    return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print(
        scrape_website_content(
            "https://medium.com/@lcrk18/how-to-optimize-sql-queries-for-better-performances-e26f60ebb36e"
        )
    )
