from openai import OpenAI
from gemini import Gemini
from llmusage import LLMUsage
from dotenv import load_dotenv
import os

load_dotenv()

usage_tracker:LLMUsage = LLMUsage(
 token_quota=10_000_000,
 cost_budget=100.0,
 )

gemini_client: Gemini = Gemini(
 model_name="gemini-2.0-flash",
 api_key=os.environ.get("GEMINI_API_KEY"),
 usage_tracker=usage_tracker
 )

print(gemini_client.generate("hi"))


