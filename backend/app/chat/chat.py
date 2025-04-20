import os
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Blueprint, request, jsonify, current_app
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.exc import IntegrityError
from app.middleware import jwt_required, admin_required
from app.models import ChatSession
from app.extensions import db

class ChatUtil:
    def __init__(self):
        load_dotenv()
        self.connection_uri = os.getenv("SQLALCHEMY_DB_DUMMY_URI")
        
    def is_user_authorized(self, user_id, session_id):
        """
        Checks if a user is authorized to access a specific chat session.
        
        Args:
            user_id: UUID of the user (either UUID object or string)
            session_id: UUID of the session (either UUID object or string)
            
        Returns:
            bool: True if the user is authorized, False otherwise
        """
        try:
            # Ensure types match model definition
            session_id_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            user_id_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
        except ValueError:
            return False  # Invalid UUID format
    
        # Use .first() for safe existence check
        session_metadata = ChatSession.query.filter_by(
            user_id=user_id_uuid,
            session_id=session_id_uuid
        ).first()
        return session_metadata is not None

    def get_chat_table(self, session_id):
        """
        Gets the chat history table for a specific session.
        
        Args:
            session_id: String representation of the session UUID
            
        Returns:
            SQLChatMessageHistory: The chat history management object
        """
        message_history = SQLChatMessageHistory(
            session_id=str(session_id), 
            connection=self.connection_uri,
            table_name="ChatHistory"
        )
        return message_history
    
    def get_all_messages(self, table):
        """
        Gets all messages from a chat history table.
        
        Args:
            table: SQLChatMessageHistory instance
            
        Returns:
            list: All messages in the chat history
        """
        return table.get_messages()
