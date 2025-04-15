from ...config import DummyConfig
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
load_dotenv()
# create sync sql message history by connection_string
message_history = SQLChatMessageHistory(
    session_id='foo', connection= DummyConfig.SQLALCHEMY_DATABASE_URI
)
message_history.add_message(HumanMessage("hello"))
print(message_history.messages)
