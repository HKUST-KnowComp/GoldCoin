import re
import time
import random
import openai
from openai import OpenAI
import asyncio
from string import punctuation
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="XXXXXXXXXXXXXXXX",
    api_version="2023-12-01-preview",
    # api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net",
)

punctuation = set(punctuation)


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        rate_limit_retry_num = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # retry on all exceptions and errors
            except Exception as e:
                print(f"Try count: {rate_limit_retry_num}, Error: {e}")
                # Increment retries
                rate_limit_retry_num += 1

                # Check if max retries has been reached
                if rate_limit_retry_num > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                match = re.search(r'after (\d+) seconds', str(e))
                if match:
                    delay = int(match.group(1)) + 2
                else:                
                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                if "triggering Azure OpenAI's content management policy" in str(e):
                    break
                # Sleep for the delay
                print(f"Retry after sleeping {delay} secs")
                time.sleep(delay)

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry_with_exponential_backoff
def chat_with_backoff(**kwargs):
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)
