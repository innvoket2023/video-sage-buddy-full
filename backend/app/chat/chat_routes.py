from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Blueprint, request, jsonify, current_app
from app.middleware import jwt_required, admin_required
from app.models import ChatSession
from app.extensions import db
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.exc import IntegrityError

load_dotenv()

chat = Blueprint('chat', __name__, url_prefix="/chat")
connection_uri = os.getenv("SQLALCHEMY_DB_DUMMY_URI")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def is_user_authorized(user_id, session_id):
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

def get_chat_table(session_id):
    """
    Gets the chat history table for a specific session.
    
    Args:
        session_id: String representation of the session UUID
        
    Returns:
        SQLChatMessageHistory: The chat history management object
    """
    message_history = SQLChatMessageHistory(
        session_id=str(session_id), 
        connection=connection_uri,
        table_name="ChatHistory"
    )
    return message_history

def get_all_messages(table):
    """
    Gets all messages from a chat history table.
    
    Args:
        table: SQLChatMessageHistory instance
        
    Returns:
        list: All messages in the chat history
    """
    return table.get_messages()

@chat.route('/start', methods=["POST"])
@jwt_required  # Ensures request.user_id exists and is valid
def start():
    user_id = request.user_id  # Assuming this is already a UUID object from jwt_required
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is missing or not JSON"}), 400

    session_id_str = data.get("session_id")  # Expect string or None from JSON
    msg = data.get("msg")

    # --- Input Validation ---
    if not msg:
        return jsonify({"error": "msg is required"}), 400

    message_uuid = str(uuid.uuid4())

    human_message = HumanMessage(content=msg, additional_kwargs={"message_uuid": message_uuid})
    conversation_sequence = []
    session_id_uuid = None  # Use UUID type for DB interaction

    # --- Determine Session and Load History ---
    try:
        if not session_id_str:  # Create NEW session
            session_id_uuid = uuid.uuid4()
            session_id_str = str(session_id_uuid)  # Keep string version for history table
            chat_session = ChatSession(user_id=user_id, session_id=session_id_uuid)
            db.session.add(chat_session)
            db.session.commit()
            # History is empty for a new session
            conversation_sequence = []
        else:  # Use EXISTING session
            # Validate session_id format before using it
            try:
                session_id_uuid = uuid.UUID(session_id_str)
            except ValueError:
                return jsonify({"error": "Invalid session_id format"}), 400

            # Check authorization (uses corrected helper with .first())
            if not is_user_authorized(user_id=user_id, session_id=session_id_uuid):
                # Changed error code to 403
                return jsonify({"error": "Not authorized to access this session_id or session not found"}), 403

            # Load existing history
            table = get_chat_table(session_id_str)  # Use string version for history table
            conversation_sequence = get_all_messages(table)

    except IntegrityError as e:  # Catch potential DB unique constraint errors
        db.session.rollback()
        current_app.logger.error(f"Database integrity error during session handling: {e}")
        return jsonify({"error": "Failed to initialize session due to database conflict"}), 500
    except Exception as e:  # Catch other potential errors (DB connection, etc.)
        db.session.rollback()  # Rollback any potential partial DB changes
        current_app.logger.error(f"Error during session initialization/history loading: {e}")
        return jsonify({"error": "Failed to initialize session or load history"}), 500

    # --- LLM Interaction & Update ---
    try:
        # Get table instance (needed for both new and existing paths)
        table = get_chat_table(session_id_str)

        # Invoke LLM (pass full history + new message for context)
        # Ensure your LLM function can handle an empty list for new chats
        llm_input = conversation_sequence + [human_message]
        llm_response = llm.invoke(llm_input)
        # Make sure response content exists
        bot_content = getattr(llm_response, 'content', 'Sorry, I could not process that.')
        bot_message = AIMessage(content=bot_content)

        # --- CORRECT History Update ---
        # Add ONLY the new human message and the new bot message
        table.add_messages([human_message, bot_message])

        # --- Return Success Response ---
        return jsonify({
            "session_id": session_id_str,  # Return the ID (useful for client)
            "response": {
                "role": "assistant",
                "content": bot_message.content
            }
        }), 200

    except Exception as e:
        # Log LLM errors or history saving errors
        current_app.logger.error(f"Error during LLM invocation or history update for session {session_id_str}: {e}")
        return jsonify({"error": "Failed to get response from AI or save history"}), 500

@chat.route('/history/<uuid:session_id>', methods=["GET"])
@jwt_required
def get_history(session_id):
    user_id = request.user_id
    session_id_str = str(session_id)

    # Check authorization
    if not is_user_authorized(user_id=user_id, session_id=session_id):
        return jsonify({"error": "Not authorized to access this session_id or session not found"}), 403

    try:
        # Get chat history
        table = get_chat_table(session_id_str)
        messages = get_all_messages(table)

        # Format messages for the response
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "user" if msg.type == "human" else "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.content,
                    "message_id": msg.additional_kwargs["message_uuid"] if role == "user" else None
                })

        return jsonify({
            "session_id": session_id_str,
            "messages": formatted_messages
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error retrieving chat history for session {session_id_str}: {e}")
        return jsonify({"error": "Failed to retrieve chat history"}), 500

@chat.route('/sessions', methods=["GET"])
@jwt_required
def get_user_sessions():
    user_id = request.user_id

    try:
        # Get all sessions for the user
        user_sessions = ChatSession.query.filter_by(user_id=user_id).order_by(
            ChatSession.updated_at.desc()
        ).all()

        # Format sessions for the response
        sessions_list = []
        for session in user_sessions:
            sessions_list.append({
                "session_id": str(session.session_id),
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            })

        return jsonify({
            "sessions": sessions_list
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error retrieving sessions for user {user_id}: {e}")
        return jsonify({"error": "Failed to retrieve user sessions"}), 500

@chat.route('/session/<uuid:session_id>', methods=["DELETE"])
@jwt_required
def delete_session(session_id):
    user_id = request.user_id

    # Check authorization
    if not is_user_authorized(user_id=user_id, session_id=session_id):
        return jsonify({"error": "Not authorized to delete this session or session not found"}), 403

    try:
        # Delete the session
        session_to_delete = ChatSession.query.filter_by(
            user_id=user_id,
            session_id=session_id
        ).first()

        if not session_to_delete:
            return jsonify({"error": "Session not found"}), 404

        # Note: This only deletes the metadata in ChatSession
        # If you want to also delete the messages, you'll need to handle that separately
        # with the SQLChatMessageHistory API
        db.session.delete(session_to_delete)
        db.session.commit()

        # Optional: Delete the messages from the history table
        try:
            table = get_chat_table(str(session_id))
            # Check if SQLChatMessageHistory has a method to delete all messages
            # If not, you might need a custom solution
            if hasattr(table, 'clear'):
                table.clear()
        except Exception as e:
            current_app.logger.warning(f"Failed to clear message history for session {session_id}: {e}")
            # Don't return an error if only metadata was deleted successfully

        return jsonify({
            "message": "Session deleted successfully",
            "session_id": str(session_id)
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"error": "Failed to delete session"}), 500

@chat.route('/sessions/<uuid:session_id>/messages/last', methods=["DELETE"])
@jwt_required
def delete_last_message(session_id):
    user_id = request.user_id
    session_id_str = str(session_id)

    # --- Authorization ---
    if not is_user_authorized(user_id=user_id, session_id=session_id):
        return jsonify({"error": "Not authorized to modify this session or session not found"}), 403

    try:
        # --- Get History ---
        table = get_chat_table(session_id_str)
        messages = get_all_messages(table)

        if not messages:
            return jsonify({"message": "No messages to delete"}), 200 # Or 404 if preferred

        # --- Identify Messages to Remove ---
        # Common case: remove last Human message and the AI message that followed it
        num_to_remove = 0
        if len(messages) >= 1 and messages[-1].type == 'ai':
            if len(messages) >= 2 and messages[-2].type == 'human':
                num_to_remove = 2 # Remove last AI and the preceding Human
            else:
                # Last message is AI, but previous wasn't Human? Edge case.
                # Decide policy: only remove AI? Error? Let's remove just the AI for now.
                num_to_remove = 1
        elif len(messages) >= 1 and messages[-1].type == 'human':
             num_to_remove = 1 # Only a human message at the end

        if num_to_remove == 0:
             return jsonify({"message": "Could not identify a user message to delete at the end"}), 400

        # Create the list *without* the messages to be deleted
        messages_to_keep = messages[:-num_to_remove]

        # --- Update History in DB (Clear and Re-add) ---
        table.clear() # Delete all messages for this session in DB
        if messages_to_keep: # Only add back if there's anything left
             table.add_messages(messages_to_keep) # Add the shortened list back

        current_app.logger.info(f"Deleted last {num_to_remove} message(s) for session {session_id_str}")
        return jsonify({
            "message": f"Successfully deleted last {num_to_remove} message(s)",
            "session_id": session_id_str
         }), 200

    except Exception as e:
        # Log DB errors or history access errors
        current_app.logger.error(f"Error deleting last message for session {session_id_str}: {e}")
        # Avoid rollback here unless you have other DB ops that might fail
        return jsonify({"error": "Failed to delete last message"}), 500

@chat.route('/sessions/<uuid:session_id>/messages/<uuid:message_uuid>', methods=["PATCH"])
@jwt_required
def edit_message(session_id, message_uuid):
    user_id = request.user_id
    session_id_str = str(session_id)
    message_uuid_str = str(message_uuid) # Assuming we get UUID from URL

    data = request.get_json()
    new_content = data.get('edited_text')

    if not new_content:
        return jsonify({"error": "new_content is required"}), 400

    # --- Authorization ---
    if not is_user_authorized(user_id=user_id, session_id=session_id):
        return jsonify({"error": "Not authorized to modify this session or session not found"}), 403

    try:
        # --- Get History ---
        table = get_chat_table(session_id_str)
        # IMPORTANT: Assumes get_all_messages retrieves additional_kwargs including 'message_uuid'
        messages = get_all_messages(table)

        edited_message_index = -1
        target_message = None

        # --- Find the Message to Edit (using assumed message_uuid) ---
        for i, msg in enumerate(messages):
            # This depends entirely on how/if you stored the custom UUID
            msg_id_from_kwargs = msg.additional_kwargs.get("message_uuid")
            if msg_id_from_kwargs == message_uuid_str:
                if msg.type != 'human':
                     return jsonify({"error": "Cannot edit AI messages"}), 403
                target_message = msg
                edited_message_index = i
                break

        if edited_message_index == -1:
            return jsonify({"error": "Message with specified UUID not found in this session"}), 404

        # --- Prepare Updated History & Regenerate ---
        # 1. Modify the message content IN THE PYTHON LIST
        messages[edited_message_index] = HumanMessage(
             content=new_content,
             additional_kwargs=target_message.additional_kwargs # Keep original kwargs like ID
        )

        # 2. Create the history slice up to the edited message
        history_up_to_edit = messages[:edited_message_index + 1]

        # 3. Clear current DB history for this session
        table.clear()

        # 4. Add back the history up to the edit point
        if history_up_to_edit:
            table.add_messages(history_up_to_edit)

        # 5. Invoke LLM with the corrected history context
        llm_response = llm.invoke(history_up_to_edit)
        bot_content = getattr(llm_response, 'content', 'Sorry, failed to regenerate response.')
        # Generate a NEW UUID for the new AI message
        new_ai_message_uuid = str(uuid.uuid4())
        new_bot_message = AIMessage(
            content=bot_content,
            additional_kwargs={"message_uuid": new_ai_message_uuid} # Store its ID too
        )

        # 6. Add the NEW AI response to the DB history
        table.add_messages([new_bot_message])

        current_app.logger.info(f"Edited message {message_uuid_str} and regenerated response for session {session_id_str}")

        # --- Return the NEW AI response ---
        # Frontend needs to know how to handle the fact that subsequent messages are now different
        return jsonify({
            "message": "Message edited and conversation continued",
            "session_id": session_id_str,
            "new_response": {
                 "role": "assistant",
                 "content": new_bot_message.content,
                 "message_uuid": new_ai_message_uuid # Send back the ID of the new AI msg
            }
            # Consider if you need to send back more context / the full updated tail?
         }), 200

    except Exception as e:
        # Log errors
        current_app.logger.error(f"Error editing message {message_uuid_str} for session {session_id_str}: {e}")
        # Avoid rollback unless necessary
        return jsonify({"error": "Failed to edit message or regenerate response"}), 500


