# from openai import OpenAI
# from gemini import Gemini
# from llmusage import LLMUsage
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# usage_tracker:LLMUsage = LLMUsage(
#  token_quota=10_000_000,
#  cost_budget=100.0,
#  )
#
# gemini_client: Gemini = Gemini(
#  model_name="gemini-2.0-flash",
#  api_key=os.environ.get("GEMINI_API_KEY"),
#  usage_tracker=usage_tracker
#  )
#
# print(gemini_client.generate("hi"))
#
import contextlib
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, save
import os
import random
import json

load_dotenv()

api = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api)

def elevenlabs_usage():

    try:
        print("\nFetching subscription information...")
        # Get subscription details
        subscription_info = client.user.get_subscription()
        something = json.dumps(str(subscription_info))
        print(something)
        # --- 1. Text-to-Speech Credits ---
        # character_count = subscription_info.character_count
        # character_limit = subscription_info.character_limit
        # remaining_characters = character_limit - character_count
        # print(f"TTS Credits Used: {character_count}")
        # print(f"TTS Credit Limit: {character_limit}")
        # print(f"TTS Credits Remaining: {remaining_characters}")
        #
        # # --- 2. Voice Clone Slots ---
        # voice_slots_available = subscription_info
        # voice_slots_used = subscription_info.voice_slots_used
        # print(f"\nVoice Clone Slots Used: {voice_slots_used}")
        #
        # # --- 3. Subscription Tier and Status ---
        # subscription_tier = subscription_info.tier
        # subscription_status = subscription_info.status
        # print(f"\nSubscription Tier: {subscription_tier}")
        # print(f"Subscription Status: {subscription_status}")
        #
        # # Optional: Print more details if needed
        # # print(f"\nFull Subscription Details:\n{subscription_info}")
    
    except Exception as e:
        print(f"An error occurred while fetching subscription info: {e}")

elevenlabs_usage()
