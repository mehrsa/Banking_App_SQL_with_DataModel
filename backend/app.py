import os
import urllib.parse
import uuid
from datetime import datetime
import json
import time
from dateutil.relativedelta import relativedelta
import numpy as np

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_sqlserver import SQLServer_VectorStore

# --- Environment and App Initialization ---
load_dotenv(override=True)
app = Flask(__name__)
CORS(app)

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
    print("⚠️  Warning: One or more Azure OpenAI environment variables are not set.")
    ai_client = None
    embeddings_client = None
else:
    ai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-15-preview",
    )
    embeddings_client = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version="2024-02-15-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
    )


# --- Database Configuration ---
server = os.getenv('DB_SERVER')
database = os.getenv('DB_DATABASE')
driver = os.getenv('DB_DRIVER', 'ODBC Driver 18 for SQL Server')
client_id = os.getenv('AZURE_CLIENT_ID')
client_secret = os.getenv('AZURE_CLIENT_SECRET')

if not all([server, database, driver, client_id, client_secret]):
    raise ValueError("Database environment variables for Service Principal are not fully configured.")

connection_string = (
    f"DRIVER={{{driver}}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={client_id};"
    f"PWD={client_secret};"
    "Authentication=ActiveDirectoryServicePrincipal;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)
sqlalchemy_url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}"
app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Vector Store Initialization ---
vector_store = None
if embeddings_client:
    vector_store = SQLServer_VectorStore(
        connection_string=sqlalchemy_url,
        table_name="DocsChunks_Embeddings",
        embedding_function=embeddings_client,
        embedding_length=1536, # Added the required embedding length
        distance_strategy=DistanceStrategy.COSINE,
    )

# --- Database Models ---
# Helper function to convert model instances to dictionaries
def to_dict_helper(instance):
    d = {}
    for column in instance.__table__.columns:
        value = getattr(instance, column.name)
        if isinstance(value, datetime):
            d[column.name] = value.isoformat()
        else:
            d[column.name] = value
    return d

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"user_{uuid.uuid4()}")
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    accounts = db.relationship('Account', backref='user', lazy=True)

    def to_dict(self):
        return to_dict_helper(self)

class Account(db.Model):
    __tablename__ = 'accounts'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"acc_{uuid.uuid4()}")
    user_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)
    account_number = db.Column(db.String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4().int)[:12])
    account_type = db.Column(db.String(50), nullable=False)
    balance = db.Column(db.Float, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"txn_{uuid.uuid4()}")
    from_account_id = db.Column(db.String(255), db.ForeignKey('accounts.id'))
    to_account_id = db.Column(db.String(255), db.ForeignKey('accounts.id'))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    category = db.Column(db.String(255))
    status = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

# Chat History and Tool Usage Models (V1)
class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"msg_{uuid.uuid4()}")
    session_id = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)
    message_type = db.Column(db.String(50), nullable=False)  # 'human', 'ai', 'system', 'tool_call', 'tool_result'
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # LangChain specific fields
    additional_kwargs = db.Column(db.JSON, default=lambda: {})
    response_md = db.Column(db.JSON, default=lambda: {})
    
    # Tool usage fields
    tool_call_id = db.Column(db.String(255))
    tool_name = db.Column(db.String(255))
    tool_input = db.Column(db.JSON)
    tool_output = db.Column(db.JSON)
    tool_error = db.Column(db.Text)
    tool_execution_time_ms = db.Column(db.Integer)

    def to_dict(self):
        return to_dict_helper(self)

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    session_id = db.Column(db.String(255), primary_key=True, default=lambda: f"session_{uuid.uuid4()}")
    user_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

class ToolUsage(db.Model):
    __tablename__ = 'tool_usage'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"tool_{uuid.uuid4()}")
    session_id = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)
    message_id = db.Column(db.String(255), db.ForeignKey('chat_history.id'))
    tool_call_id = db.Column(db.String(255), nullable=False)
    tool_name = db.Column(db.String(255), nullable=False)
    tool_input = db.Column(db.JSON, nullable=False)
    tool_output = db.Column(db.JSON)
    tool_error = db.Column(db.Text)
    execution_time_ms = db.Column(db.Integer)
    status = db.Column(db.String(50), default='pending')  # 'pending', 'success', 'error', 'timeout'
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    
    # Additional tracking fields
    cost_cents = db.Column(db.Integer)  # For paid APIs
    tokens_used = db.Column(db.Integer)
    rate_limit_hit = db.Column(db.Boolean, default=False)
    retry_count = db.Column(db.Integer, default=0)

    def to_dict(self):
        return to_dict_helper(self)

class ToolDefinition(db.Model):
    __tablename__ = 'tool_definitions'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"tooldef_{uuid.uuid4()}")
    name = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.Text)
    input_schema = db.Column(db.JSON, nullable=False)
    version = db.Column(db.String(50), default='1.0.0')
    is_active = db.Column(db.Boolean, default=True)
    cost_per_call_cents = db.Column(db.Integer, default=0)
    average_execution_time_ms = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

# --- Chat History Management Class ---
class ChatHistoryManager:
    def __init__(self, session_id: str, user_id: str = 'user_1'):
        self.session_id = session_id
        self.user_id = user_id
        self._ensure_session_exists()

    def _ensure_session_exists(self):
        """Ensure the chat session exists in the database"""
        session = ChatSession.query.filter_by(session_id=self.session_id).first()
        if not session:
            session = ChatSession(
                session_id=self.session_id,
                user_id=self.user_id,
                title="New Chat Session"
            )
            print("-----------------> New chat session created: ", session.session_id)
            db.session.add(session)
            db.session.commit()

    def add_message(self, message_type: str, content: str, **kwargs):
        """Add a message to the chat history"""
        message = ChatHistory(
            session_id=self.session_id,
            user_id=self.user_id,
            message_type=message_type,
            content=content,
            **kwargs
        )
        db.session.add(message)
        db.session.commit()
        return message

    def add_tool_call(self, tool_call_id: str, tool_name: str, tool_input: dict, content: str = None):
        """Log a tool call"""
        content = content or f"Calling tool: {tool_name}"
        print(f"Adding tool call: {tool_name} with ID: {tool_call_id}")
        return self.add_message(
            message_type='tool_call',
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input
        )

    def add_tool_result(self, tool_call_id: str, tool_name: str, tool_output: dict, 
                       content: str = None, error: str = None, execution_time_ms: int = None):
        """Log a tool result"""
        content = content or f"Tool {tool_name} result"
        return self.add_message(
            message_type='tool_result',
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_output=tool_output,
            tool_error=error,
            tool_execution_time_ms=execution_time_ms
        )

    def log_tool_usage(self, tool_call_id: str, tool_name: str, tool_input: dict, 
                      tool_output: dict = None, error: str = None, execution_time_ms: int = None, 
                      message_id: str = None, tokens_used: int = None):
        """Log detailed tool usage metrics"""
        status = 'error' if error else 'success'
        
        tool_usage = ToolUsage(
            session_id=self.session_id,
            user_id=self.user_id,
            message_id=message_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            tool_error=error,
            execution_time_ms=execution_time_ms,
            status=status,
            completed_at=datetime.utcnow(),
            tokens_used=tokens_used
        )
        db.session.add(tool_usage)
        db.session.commit()
        return tool_usage

    def get_conversation_history(self, limit: int = 50):
        """Retrieve conversation history for this session"""
        messages = ChatHistory.query.filter_by(
            session_id=self.session_id
        ).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
        
        return [msg.to_dict() for msg in reversed(messages)]

# --- AI Chatbot Tool Definitions (Enhanced with logging) ---

def get_user_accounts(user_id='user_1'):
    """Retrieves all accounts for a given user."""
    try:
        accounts = Account.query.filter_by(user_id=user_id).all()
        if not accounts:
            return "No accounts found for this user."
        return json.dumps([
            {"name": acc.name, "account_type": acc.account_type, "balance": acc.balance} 
            for acc in accounts
        ])
    except Exception as e:
        return f"Error retrieving accounts: {str(e)}"

def get_transactions_summary(user_id='user_1', time_period='this month', account_name=None):
    """Provides a summary of the user's spending. Can be filtered by a time period and a specific account."""
    try:
        query = db.session.query(Transaction.category, db.func.sum(Transaction.amount).label('total_spent')).filter(
            Transaction.type == 'payment'
        )
        if account_name:
            account = Account.query.filter_by(user_id=user_id, name=account_name).first()
            if not account:
                return json.dumps({"status": "error", "message": f"Account '{account_name}' not found."})
            query = query.filter(Transaction.from_account_id == account.id)
        else:
            user_accounts = Account.query.filter_by(user_id=user_id).all()
            account_ids = [acc.id for acc in user_accounts]
            query = query.filter(Transaction.from_account_id.in_(account_ids))

        end_date = datetime.utcnow()
        if 'last 6 months' in time_period.lower():
            start_date = end_date - relativedelta(months=6)
        elif 'this year' in time_period.lower():
            start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0)
        else:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0)
        
        query = query.filter(Transaction.created_at.between(start_date, end_date))
        results = query.group_by(Transaction.category).order_by(db.func.sum(Transaction.amount).desc()).all()
        total_spending = sum(r.total_spent for r in results)
        
        summary_details = {
            "total_spending": round(total_spending, 2),
            "period": time_period,
            "account_filter": account_name or "All Accounts",
            "top_categories": [{"category": r.category, "amount": round(r.total_spent, 2)} for r in results[:3]]
        }

        if not results:
            return json.dumps({"status": "success", "summary": f"You have no spending for the period '{time_period}' in account '{account_name or 'All Accounts'}'."})

        return json.dumps({"status": "success", "summary": summary_details})
    except Exception as e:
        print(f"ERROR in get_transactions_summary: {e}")
        return json.dumps({"status": "error", "message": f"An error occurred while generating the transaction summary."})

def search_support_documents(user_question: str):
    """Searches the knowledge base for answers to customer support questions using vector search."""
    if not vector_store:
        return "The vector store is not configured."
    try:
        results = vector_store.similarity_search_with_score(user_question, k=3)
        relevant_docs = [doc.page_content for doc, score in results if score < 0.5]
        
        if not relevant_docs:
            return "No relevant support documents found to answer this question."

        context = "\n\n---\n\n".join(relevant_docs)
        return context

    except Exception as e:
        print(f"ERROR in search_support_documents: {e}")
        return "An error occurred while searching for support documents."

def create_new_account(user_id='user_1', account_type='checking', name=None, balance=0.0):
    """Creates a new bank account for the user."""
    if not name:
        return json.dumps({"status": "error", "message": "An account name is required."})
    try:
        new_account = Account(user_id=user_id, account_type=account_type, balance=balance, name=name)
        db.session.add(new_account)
        db.session.commit()
        return json.dumps({
            "status": "success", "message": f"Successfully created new {account_type} account '{name}' with balance ${balance:.2f}.",
            "account_id": new_account.id, "account_name": new_account.name
        })
    except Exception as e:
        db.session.rollback()
        return f"Error creating account: {str(e)}"

def transfer_money(user_id='user_1', from_account_name=None, to_account_name=None, amount=0.0, to_external_details=None):
    """Transfers money between user's accounts or to an external account."""
    if not from_account_name or (not to_account_name and not to_external_details) or amount <= 0:
        return json.dumps({"status": "error", "message": "Missing required transfer details."})
    try:
        from_account = Account.query.filter_by(user_id=user_id, name=from_account_name).first()
        if not from_account:
            return json.dumps({"status": "error", "message": f"Account '{from_account_name}' not found."})
        if from_account.balance < amount:
            return json.dumps({"status": "error", "message": "Insufficient funds."})
        
        to_account = None
        if to_account_name:
            to_account = Account.query.filter_by(user_id=user_id, name=to_account_name).first()
            if not to_account:
                 return json.dumps({"status": "error", "message": f"Recipient account '{to_account_name}' not found."})
        
        new_transaction = Transaction(
            from_account_id=from_account.id, to_account_id=to_account.id if to_account else None,
            amount=amount, type='transfer', description=f"Transfer to {to_account_name or to_external_details.get('name', 'External')}",
            category='Transfer', status='completed'
        )
        from_account.balance -= amount
        if to_account:
            to_account.balance += amount
        db.session.add(new_transaction)
        db.session.commit()
        return json.dumps({"status": "success", "message": f"Successfully transferred ${amount:.2f}."})
    except Exception as e:
        db.session.rollback()
        return f"Error during transfer: {str(e)}"

# --- API Routes ---
@app.route('/api/accounts', methods=['GET', 'POST'])
def handle_accounts():
    user_id = 'user_1'
    if request.method == 'GET':
        accounts = Account.query.filter_by(user_id=user_id).all()
        return jsonify([acc.to_dict() for acc in accounts])
    if request.method == 'POST':
        data = request.json
        account_str = create_new_account(user_id=user_id, account_type=data.get('account_type'), name=data.get('name'), balance=data.get('balance', 0))
        return jsonify(json.loads(account_str)), 201

@app.route('/api/transactions', methods=['GET', 'POST'])
def handle_transactions():
    user_id = 'user_1'
    if request.method == 'GET':
        accounts = Account.query.filter_by(user_id=user_id).all()
        account_ids = [acc.id for acc in accounts]
        transactions = Transaction.query.filter((Transaction.from_account_id.in_(account_ids)) | (Transaction.to_account_id.in_(account_ids))).order_by(Transaction.created_at.desc()).all()
        return jsonify([t.to_dict() for t in transactions])
    if request.method == 'POST':
        data = request.json
        result_str = transfer_money(
            user_id=user_id, from_account_name=data.get('from_account_name'), to_account_name=data.get('to_account_name'),
            amount=data.get('amount'), to_external_details=data.get('to_external_details')
        )
        result = json.loads(result_str)
        status_code = 201 if result.get("status") == "success" else 400
        return jsonify(result), status_code

# Chat History API Routes
@app.route('/api/chat/sessions', methods=['GET', 'POST'])
def handle_chat_sessions():
    user_id = 'user_1'  # In production, get from auth
    
    if request.method == 'GET':
        sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()
        return jsonify([session.to_dict() for session in sessions])
    
    if request.method == 'POST':
        data = request.json
        session = ChatSession(
            user_id=user_id,
            title=data.get('title', 'New Chat Session'),
        )
        db.session.add(session)
        db.session.commit()
        return jsonify(session.to_dict()), 201



# Added this for now to enforce one session per app launch. 
# It kept changing session id during the in progress chat (needs fixing)
global_session_id = None

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    global global_session_id
    if not ai_client:
        return jsonify({"error": "Azure OpenAI client is not configured."}), 503

    data = request.json
    messages = data.get("messages", [])
    if global_session_id is not None:
        session_id = global_session_id
    else:
        session_id = data.get("session_id") or f"session_{uuid.uuid4()}"
        global_session_id = session_id

    user_id = data.get("user_id", "user_1")
    
    # Initialize chat history manager
    chat_manager = ChatHistoryManager(session_id, user_id)
    
    # Log the user's message
    if messages and messages[-1].get("role") == "user":
        chat_manager.add_message(
            message_type='human',
            content=messages[-1].get("content")
        )

    tools = [
        {"type": "function", "function": {
            "name": "get_user_accounts",
            "description": "Get a list of all bank accounts belonging to the current user.",
            "parameters": {"type": "object", "properties": {}}
        }},
        {"type": "function", "function": {
            "name": "get_transactions_summary",
            "description": "Get a summary of spending for the user, filterable by account and time period.",
             "parameters": {
                "type": "object",
                "properties": {
                    "time_period": {"type": "string", "description": "e.g., 'this month', 'last 6 months'."},
                    "account_name": {"type": "string", "description": "e.g., 'Primary Checking'."}
                },
            }
        }},
        {"type": "function", "function": {
            "name": "search_support_documents",
            "description": "Use this for customer support questions, such as 'how to do X', 'what are the fees for Y', or policy questions.",
             "parameters": {
                "type": "object",
                "properties": {"user_question": {"type": "string", "description": "The user's full question."}},
                "required": ["user_question"]
            }
        }},
        {"type": "function", "function": {
            "name": "create_new_account",
            "description": "Creates a new bank account for the user.",
            "parameters": {"type": "object", "properties": {
                "account_type": {"type": "string", "enum": ["checking", "savings", "credit"]},
                "name": {"type": "string", "description": "The desired name for the new account."},
                "balance": {"type": "number", "description": "The initial balance."}}, "required": ["account_type", "name"]
            }
        }},
        {"type": "function", "function": {
            "name": "transfer_money",
            "description": "Transfer funds between accounts or to an external account.",
            "parameters": {"type": "object", "properties": {
                "from_account_name": {"type": "string"}, "to_account_name": {"type": "string"}, "amount": {"type": "number"},
                "to_external_details": {"type": "object", "properties": {"name": {"type": "string"},"accountNumber": {"type": "string"},"routingNumber": {"type": "string"}}}
                }, "required": ["from_account_name", "amount"]
            }
        }}
    ]

    response = ai_client.chat.completions.create(model=AZURE_OPENAI_DEPLOYMENT, messages=messages, tools=tools, tool_choice="auto")
        

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls


    if tool_calls:
        messages.append(response_message)
        available_functions = {
            "get_user_accounts": get_user_accounts,
            "get_transactions_summary": get_transactions_summary,
            "search_support_documents": search_support_documents,
            "create_new_account": create_new_account,
            "transfer_money": transfer_money,
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Log tool call start
            chat_manager.add_tool_call(
                tool_call_id=tool_call.id,
                tool_name=function_name,
                tool_input=function_args
            )
            
            # Execute the tool
            tool_start_time = time.time()
            function_to_call = available_functions[function_name]
            
            try:
                function_response = function_to_call(**function_args)
                tool_execution_time = int((time.time() - tool_start_time) * 1000)
                tool_error = None
                tool_output = {"result": function_response}
                
            except Exception as e:
                tool_execution_time = int((time.time() - tool_start_time) * 1000)
                tool_error = str(e)
                function_response = f"Error executing {function_name}: {str(e)}"
                tool_output = {"error": str(e)}
            
            # Log tool result in chat history
            tool_result_message = chat_manager.add_tool_result(
                tool_call_id=tool_call.id,
                tool_name=function_name,
                tool_output=tool_output,
                content=function_response[:500] + "..." if len(str(function_response)) > 500 else str(function_response),
                error=tool_error,
                execution_time_ms=tool_execution_time
            )
            
            # Log detailed tool usage metrics
            chat_manager.log_tool_usage(
                tool_call_id=tool_call.id,
                tool_name=function_name,
                tool_input=function_args,
                tool_output=tool_output,
                error=tool_error,
                execution_time_ms=tool_execution_time,
                tokens_used=response.usage.total_tokens,
                message_id=tool_result_message.id
            )
            
            # Prepare content for LangChain message format
            tool_message_content = function_response
            if function_name == 'search_support_documents':
                rag_instruction = "You are a customer support agent. Answer the user's last question based *only* on the following document context. If the context says no documents were found, inform the user you could not find an answer. Do not use your general knowledge. CONTEXT: "
                tool_message_content = rag_instruction + function_response

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_message_content,
            })
        
        # Second API call to get a natural language response based on the tool's output
        final_start_time = time.time()
        second_response = ai_client.chat.completions.create(model=AZURE_OPENAI_DEPLOYMENT, messages=messages)
        final_response_time = int((time.time() - final_start_time) * 1000)
        
        final_message = second_response.choices[0].message.content
        
        # Log the final AI response
        chat_manager.add_message(
            message_type='ai',
            content=final_message,
            response_md={
                "model": AZURE_OPENAI_DEPLOYMENT,
                "response_time_ms": final_response_time,
                "tool_calls_made": len(tool_calls)
            }
        )
        
        return jsonify({
            "response": final_message,
            "session_id": session_id,
            "tools_used": [tc.function.name for tc in tool_calls]
        })

    # If no tool is called, just return the model's direct response
    return jsonify({
        "response": response_message.content,
        "session_id": session_id,
        "tools_used": []
    })

# Tool Management Routes
@app.route('/api/tools/definitions', methods=['GET', 'POST'])
def handle_tool_definitions():
    if request.method == 'GET':
        tools = ToolDefinition.query.filter_by(is_active=True).all()
        return jsonify([tool.to_dict() for tool in tools])
    
    if request.method == 'POST':
        data = request.json
        tool_def = ToolDefinition(
            name=data['name'],
            description=data.get('description'),
            input_schema=data['input_schema'],
            version=data.get('version', '1.0.0'),
            cost_per_call_cents=data.get('cost_per_call_cents', 0)
        )
        db.session.add(tool_def)
        db.session.commit()
        return jsonify(tool_def.to_dict()), 201

@app.route('/api/tools/usage/<session_id>', methods=['GET'])
def get_session_tool_usage(session_id):
    """Get tool usage for a specific session"""
    usage = ToolUsage.query.filter_by(session_id=session_id).order_by(ToolUsage.started_at.desc()).all()
    return jsonify([u.to_dict() for u in usage])

@app.route('/api/chat/export/<session_id>', methods=['GET'])
def export_chat_session(session_id):
    """Export a complete chat session with tool usage"""
    chat_manager = ChatHistoryManager(session_id)
    history = chat_manager.get_conversation_history(limit=1000)
    
    # Get tool usage for this session
    tool_usage = ToolUsage.query.filter_by(session_id=session_id).all()
    
    export_data = {
        "session_id": session_id,
        "exported_at": datetime.utcnow().isoformat(),
        "chat_history": history,
        "tool_usage": [usage.to_dict() for usage in tool_usage],
        "summary": {
            "total_messages": len(history),
            "total_tool_calls": len(tool_usage),
            "unique_tools_used": len(set(usage.tool_name for usage in tool_usage)),
            "average_tool_execution_time": np.mean([usage.execution_time_ms for usage in tool_usage if usage.execution_time_ms]) if tool_usage else 0
        }
    }
    
    return jsonify(export_data)

# --- Database Initialization ---
def initialize_tool_definitions():
    """Initialize tool definitions in the database"""
    tools_data = [
        {
            "name": "get_user_accounts",
            "description": "Retrieves all accounts for a given user",
            "input_schema": {"type": "object", "properties": {}},
            "cost_per_call_cents": 0
        },
        {
            "name": "get_transactions_summary",
            "description": "Provides spending summary with time period and account filters",
            "input_schema": {
                "type": "object",
                "properties": {
                    "time_period": {"type": "string"},
                    "account_name": {"type": "string"}
                }
            },
            "cost_per_call_cents": 0
        },
        {
            "name": "search_support_documents",
            "description": "Searches knowledge base for customer support answers",
            "input_schema": {
                "type": "object",
                "properties": {"user_question": {"type": "string"}},
                "required": ["user_question"]
            },
            "cost_per_call_cents": 2  # Embedding search has slight cost (dummy value for now)
        },
        {
            "name": "create_new_account",
            "description": "Creates a new bank account for the user",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_type": {"type": "string", "enum": ["checking", "savings", "credit"]},
                    "name": {"type": "string"},
                    "balance": {"type": "number"}
                },
                "required": ["account_type", "name"]
            },
            "cost_per_call_cents": 0
        },
        {
            "name": "transfer_money",
            "description": "Transfers money between accounts or to external accounts",
            "input_schema": {
                "type": "object",
                "properties": {
                    "from_account_name": {"type": "string"},
                    "to_account_name": {"type": "string"},
                    "amount": {"type": "number"},
                    "to_external_details": {"type": "object"}
                },
                "required": ["from_account_name", "amount"]
            },
            "cost_per_call_cents": 0
        }
    ]
    
    for tool_data in tools_data:
        existing_tool = ToolDefinition.query.filter_by(name=tool_data["name"]).first()
        if not existing_tool:
            tool_def = ToolDefinition(**tool_data)
            db.session.add(tool_def)
    
    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        initialize_tool_definitions()
    app.run(debug=True, port=5001)