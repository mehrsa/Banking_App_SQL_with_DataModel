-- Use this to create the core chat history tables:

-- 1. Chat Sessions Table to manage individual conversation sessions
CREATE TABLE chat_sessions (
    session_id NVARCHAR(255) PRIMARY KEY,
    user_id NVARCHAR(255) NOT NULL,
    title NVARCHAR(500),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE()
);

-- 2. Chat History Table for storing all conversation messages and tool interactions
CREATE TABLE chat_history (
    id NVARCHAR(255) PRIMARY KEY,
    session_id NVARCHAR(255) NOT NULL,
    user_id NVARCHAR(255) NOT NULL,
    message_type NVARCHAR(50) NOT NULL, -- 'human', 'ai', 'system', 'tool_call', 'tool_result'
    content NVARCHAR(MAX),
    timestamp DATETIME2 DEFAULT GETUTCDATE(),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    
    -- LangChain specific fields
    additional_kwargs NVARCHAR(MAX) DEFAULT '{}', -- JSON data for LangChain message kwargs
    response_metadata NVARCHAR(MAX) DEFAULT '{}', -- JSON data for AI response metadata
    
    -- Tool usage fields (embedded in message)
    tool_call_id NVARCHAR(255),
    tool_name NVARCHAR(255),
    tool_input NVARCHAR(MAX), -- JSON data for tool input parameters
    tool_output NVARCHAR(MAX), -- JSON data for tool output/results
    tool_error NVARCHAR(MAX), -- Text description of any tool errors
    tool_execution_time_ms INT -- Execution time in milliseconds
);

-- 3. Tool Usage Table for detailed metrics and tracking for individual tool executions
CREATE TABLE tool_usage (
    id NVARCHAR(255) PRIMARY KEY,
    session_id NVARCHAR(255) NOT NULL,
    user_id NVARCHAR(255) NOT NULL,
    message_id NVARCHAR(255), -- Links to chat_history.id
    tool_call_id NVARCHAR(255) NOT NULL,
    tool_name NVARCHAR(255) NOT NULL,
    tool_input NVARCHAR(MAX) NOT NULL, -- JSON data for input parameters
    tool_output NVARCHAR(MAX), -- JSON data for output/results
    tool_error NVARCHAR(MAX), -- Error message if tool failed
    execution_time_ms INT, -- Execution time in milliseconds
    status NVARCHAR(50) DEFAULT 'pending', -- 'pending', 'success', 'error', 'timeout'
    started_at DATETIME2 DEFAULT GETUTCDATE(),
    completed_at DATETIME2,
    
    -- Additional tracking fields
    cost_cents INT, -- Cost in cents for paid API tools
    tokens_used INT, -- Number of tokens consumed
    rate_limit_hit BIT DEFAULT 0, -- Whether rate limit was encountered
    retry_count INT DEFAULT 0 -- Number of retry attempts
);

-- 4. Tool Definitions Table capturing all available tools with their schemas and metadata
CREATE TABLE tool_definitions (
    id NVARCHAR(255) PRIMARY KEY,
    name NVARCHAR(255) UNIQUE NOT NULL,
    description NVARCHAR(MAX),
    input_schema NVARCHAR(MAX) NOT NULL, -- JSON Schema for input validation
    version NVARCHAR(50) DEFAULT '1.0.0',
    is_active BIT DEFAULT 1,
    cost_per_call_cents INT DEFAULT 0,
    average_execution_time_ms INT,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE()
);


-- Foreign Key Constraints
-- Link chat_history to chat_sessions
ALTER TABLE chat_history 
ADD CONSTRAINT FK_chat_history_session 
FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id);

-- Link tool_usage to chat_sessions
ALTER TABLE tool_usage 
ADD CONSTRAINT FK_tool_usage_session 
FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id);

-- Link tool_usage to chat_history (optional relationship)
ALTER TABLE tool_usage 
ADD CONSTRAINT FK_tool_usage_message 
FOREIGN KEY (message_id) REFERENCES chat_history(id);
