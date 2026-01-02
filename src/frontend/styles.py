"""Custom CSS styles for the chat interface."""

def get_chat_styles() -> str:
    """Return CSS styles for the chat interface."""
    return """
    <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        
        /* Chat Interface Container - Simple fixed positioning */
        .chat-container {
            position: fixed !important;
            bottom: 20px !important;
            right: 20px !important;
            width: 400px !important;
            max-height: 600px !important;
            background: white !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #e0e0e0 !important;
            z-index: 1000 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Chat header */
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 12px 12px 0 0;
            font-weight: 600;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Messages container */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background: #fafafa;
            max-height: 400px;
        }
        
        /* Message bubbles - User vs Assistant distinction */
        .message-bubble {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        
        .message-bubble.user {
            flex-direction: row-reverse;
        }
        
        .message-content {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.9rem;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        /* User message styling */
        .message-content.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 6px;
        }
        
        /* Assistant message styling */
        .message-content.assistant {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 6px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Message avatar */
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        
        .message-avatar.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message-avatar.assistant {
            background: #f0f0f0;
            color: #666;
        }
        
        /* Message timestamp */
        .message-timestamp {
            font-size: 0.7rem;
            color: #888;
            margin-top: 4px;
            text-align: right;
        }
        
        .message-bubble.user .message-timestamp {
            text-align: left;
        }
        
        /* Input area styling */
        .chat-input-container {
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            border-radius: 0 0 12px 12px;
        }
        
        /* Streamlit chat input customization */
        .stChatInput > div {
            background-color: #f8f9fa !important;
            border: 1px solid #ddd !important;
            border-radius: 20px !important;
            padding: 8px 15px !important;
        }
        
        .stChatInput input {
            border: none !important;
            background: transparent !important;
            font-size: 0.9rem !important;
        }
        
        .stChatInput input:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Ensure chat input appears above other elements */
        .stChatInput {
            position: relative !important;
            z-index: 1001 !important;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 18px;
            border-bottom-left-radius: 6px;
            max-width: 75%;
            margin-bottom: 15px;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dot {
            width: 6px;
            height: 6px;
            background: #999;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        /* Collapsible chat toggle button */
        .chat-toggle-button {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
            z-index: 1000;
            transition: all 0.3s ease;
        }
        
        .chat-toggle-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
        }
        
        /* Responsive behavior */
        @media (max-width: 768px) {
            .chat-container {
                width: 100%;
                max-width: 400px;
                margin: 0 auto;
                position: relative;
                top: 0;
            }
            
            .chat-toggle-button {
                bottom: 20px;
                right: 20px;
                width: 50px;
                height: 50px;
                font-size: 1.2rem;
            }
            
            .message-content {
                max-width: 85%;
                font-size: 0.85rem;
            }
            
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        
        @media (max-width: 480px) {
            .chat-container {
                width: 100%;
                min-height: 400px;
                border-radius: 8px;
            }
            
            .chat-header {
                padding: 12px 15px;
                font-size: 1rem;
                border-radius: 8px 8px 0 0;
            }
            
            .messages-container {
                padding: 10px;
                max-height: 300px;
            }
            
            .message-content {
                padding: 10px 12px;
                font-size: 0.8rem;
                max-width: 90%;
            }
            
            .message-avatar {
                width: 28px;
                height: 28px;
                font-size: 0.9rem;
            }
            
            .chat-input-container {
                padding: 12px 15px;
            }
        }
        
        /* Loading states */
        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Error states */
        .error-message {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 12px 16px;
            border-radius: 18px;
            border-bottom-left-radius: 6px;
            max-width: 75%;
            margin-bottom: 15px;
        }
        
        /* Success states */
        .success-message {
            background: #efe;
            border: 1px solid #cfc;
            color: #3c3;
            padding: 12px 16px;
            border-radius: 18px;
            border-bottom-left-radius: 6px;
            max-width: 75%;
            margin-bottom: 15px;
        }
        
        /* Scrollbar styling for messages container */
        .messages-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .messages-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .messages-container::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .messages-container::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        
        /* Sidebar styling enhancements */
        .sidebar .element-container {
            margin-bottom: 1rem;
        }
        
        /* Button styling improvements */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            border-color: #667eea;
            color: #667eea;
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            color: white;
        }
    </style>
    """


def get_collapsible_chat_styles() -> str:
    """Return additional CSS for collapsible chat functionality."""
    return """
    <style>
        /* Collapsible chat specific styles */
        .chat-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .chat-widget.collapsed {
            width: 60px;
            height: 60px;
        }
        
        .chat-widget.expanded {
            width: 400px;
            height: 600px;
        }
        
        .chat-widget-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 12px 12px 0 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
        }
        
        .chat-widget-body {
            background: white;
            border: 1px solid #e0e0e0;
            border-top: none;
            border-radius: 0 0 12px 12px;
            height: calc(100% - 60px);
            display: flex;
            flex-direction: column;
        }
        
        .minimize-button {
            background: none;
            border: none;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: background 0.2s ease;
        }
        
        .minimize-button:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Animation for expand/collapse */
        .chat-widget {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Mobile responsive for collapsible chat */
        @media (max-width: 768px) {
            .chat-widget.expanded {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                width: 100%;
                height: 100%;
                border-radius: 0;
            }
            
            .chat-widget-header {
                border-radius: 0;
            }
            
            .chat-widget-body {
                border-radius: 0;
                height: calc(100% - 60px);
            }
        }
    </style>
    """


def get_message_bubble_styles() -> str:
    """Return CSS specifically for message bubble styling."""
    return """
    <style>
        /* Enhanced message bubble styles */
        .chat-message {
            display: flex;
            margin-bottom: 16px;
            align-items: flex-start;
            gap: 12px;
        }
        
        .chat-message.user {
            flex-direction: row-reverse;
        }
        
        .chat-message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            line-height: 1.5;
            word-wrap: break-word;
            position: relative;
        }
        
        /* User message bubble */
        .chat-message-content.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 8px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        
        /* Assistant message bubble */
        .chat-message-content.assistant {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        
        /* Message tails */
        .chat-message-content.user::after {
            content: '';
            position: absolute;
            bottom: 0;
            right: -8px;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            border-left-color: #764ba2;
            border-bottom: none;
            border-top-left-radius: 4px;
        }
        
        .chat-message-content.assistant::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: -8px;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            border-right-color: white;
            border-bottom: none;
            border-top-right-radius: 4px;
        }
        
        /* Avatar styling */
        .chat-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
            font-weight: 600;
        }
        
        .chat-avatar.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .chat-avatar.assistant {
            background: #f8f9fa;
            color: #667eea;
            border: 2px solid #e0e0e0;
        }
        
        /* Timestamp styling */
        .message-time {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 4px;
            text-align: right;
        }
        
        .chat-message.assistant .message-time {
            color: #888;
            text-align: left;
        }
        
        /* Code blocks in messages */
        .chat-message-content pre {
            background: rgba(0, 0, 0, 0.1);
            padding: 8px 12px;
            border-radius: 8px;
            margin: 8px 0;
            overflow-x: auto;
            font-size: 0.8rem;
        }
        
        .chat-message-content.user pre {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Links in messages */
        .chat-message-content a {
            color: inherit;
            text-decoration: underline;
            opacity: 0.9;
        }
        
        .chat-message-content.user a {
            color: white;
        }
        
        .chat-message-content.assistant a {
            color: #667eea;
        }
    </style>
    """


def get_input_area_styles() -> str:
    """Return CSS for input area styling."""
    return """
    <style>
        /* Input area styling */
        .chat-input-wrapper {
            padding: 16px 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            border-radius: 0 0 12px 12px;
        }
        
        .chat-input-field {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 24px;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s ease;
            background: #f8f9fa;
        }
        
        .chat-input-field:focus {
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .chat-input-field::placeholder {
            color: #999;
        }
        
        /* Send button */
        .send-button {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .send-button:hover {
            transform: translateY(-50%) scale(1.1);
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: translateY(-50%) scale(1);
        }
        
        /* Character counter */
        .char-counter {
            font-size: 0.75rem;
            color: #888;
            text-align: right;
            margin-top: 4px;
        }
        
        .char-counter.warning {
            color: #f39c12;
        }
        
        .char-counter.error {
            color: #e74c3c;
        }
        
        /* Input container positioning */
        .input-container {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        /* Streamlit chat input overrides */
        .stChatInput {
            padding: 0 !important;
        }
        
        .stChatInput > div {
            background: #f8f9fa !important;
            border: 1px solid #ddd !important;
            border-radius: 24px !important;
            padding: 0 !important;
        }
        
        .stChatInput input {
            padding: 12px 16px !important;
            border: none !important;
            background: transparent !important;
            font-size: 0.9rem !important;
        }
        
        .stChatInput input:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Focus state for input container */
        .stChatInput > div:focus-within {
            border-color: #667eea !important;
            background: white !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
    </style>
    """


def get_responsive_styles() -> str:
    """Return responsive CSS for different screen sizes."""
    return """
    <style>
        /* Responsive design for chat interface */
        
        /* Tablet styles */
        @media (max-width: 1024px) {
            .chat-container {
                width: 350px;
            }
            
            .message-content {
                max-width: 80%;
            }
        }
        
        /* Mobile landscape */
        @media (max-width: 768px) {
            .chat-interface-container,
            .chat-container {
                width: calc(100vw - 40px) !important;
                max-width: 400px !important;
                right: 20px !important;
                left: 20px !important;
                margin: 0 auto !important;
            }
            
            .chat-toggle-button {
                bottom: 20px;
                right: 20px;
                width: 50px;
                height: 50px;
                font-size: 1.2rem;
            }
            
            .message-content {
                max-width: 85%;
                font-size: 0.85rem;
                padding: 10px 14px;
            }
            
            .message-avatar {
                width: 30px;
                height: 30px;
                font-size: 0.95rem;
            }
            
            .chat-header,
            .chat-header-fixed {
                padding: 12px 16px;
                font-size: 1rem;
            }
            
            .messages-container,
            .chat-messages-fixed {
                padding: 12px;
                max-height: 300px;
            }
            
            .chat-input-container,
            .chat-input-fixed {
                padding: 12px 16px;
            }
        }
        
        /* Mobile portrait */
        @media (max-width: 480px) {
            .main .block-container {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            
            .chat-interface-container,
            .chat-container {
                width: calc(100vw - 20px) !important;
                right: 10px !important;
                left: 10px !important;
                bottom: 10px !important;
                min-height: 400px;
                border-radius: 8px;
                max-height: 60vh;
            }
            
            .chat-header,
            .chat-header-fixed {
                padding: 10px 12px;
                font-size: 0.95rem;
                border-radius: 8px 8px 0 0;
            }
            
            .messages-container,
            .chat-messages-fixed {
                padding: 8px;
                max-height: 280px;
            }
            
            .message-content {
                padding: 8px 12px;
                font-size: 0.8rem;
                max-width: 90%;
                border-radius: 16px;
            }
            
            .message-content.user {
                border-bottom-right-radius: 6px;
            }
            
            .message-content.assistant {
                border-bottom-left-radius: 6px;
            }
            
            .message-avatar {
                width: 26px;
                height: 26px;
                font-size: 0.85rem;
            }
            
            .chat-input-container,
            .chat-input-fixed {
                padding: 10px 12px;
            }
            
            .chat-input-field {
                padding: 10px 14px;
                font-size: 0.85rem;
                border-radius: 20px;
            }
            
            .send-button {
                width: 28px;
                height: 28px;
                right: 6px;
            }
            
            .chat-toggle-button {
                bottom: 15px;
                right: 15px;
                width: 45px;
                height: 45px;
                font-size: 1.1rem;
            }
        }
        
        /* Extra small screens */
        @media (max-width: 320px) {
            .chat-container {
                border-radius: 6px;
                max-height: 50vh;
            }
            
            .chat-header {
                padding: 8px 10px;
                font-size: 0.9rem;
            }
            
            .messages-container {
                padding: 6px;
                max-height: 220px;
            }
            
            .message-content {
                padding: 6px 10px;
                font-size: 0.75rem;
                max-width: 95%;
            }
            
            .message-avatar {
                width: 24px;
                height: 24px;
                font-size: 0.8rem;
            }
            
            .chat-input-container {
                padding: 8px 10px;
            }
            
            .chat-input-field {
                padding: 8px 12px;
                font-size: 0.8rem;
            }
        }
        
        /* High DPI displays */
        @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
            .message-content {
                border-width: 0.5px;
            }
            
            .chat-container {
                border-width: 0.5px;
            }
        }
        
        /* Dark mode support (if needed) */
        @media (prefers-color-scheme: dark) {
            .chat-container {
                background: #2d3748;
                border-color: #4a5568;
            }
            
            .messages-container {
                background: #1a202c;
            }
            
            .message-content.assistant {
                background: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            
            .chat-input-field {
                background: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            
            .chat-input-field::placeholder {
                color: #a0aec0;
            }
        }
        
        /* Print styles */
        @media print {
            .chat-toggle-button,
            .chat-input-container {
                display: none !important;
            }
            
            .chat-container {
                position: static !important;
                width: 100% !important;
                max-height: none !important;
                box-shadow: none !important;
                border: 1px solid #000 !important;
            }
            
            .messages-container {
                max-height: none !important;
                overflow: visible !important;
            }
        }
    </style>
    """


def get_all_chat_styles() -> str:
    """Return all chat interface styles combined."""
    return (
        get_chat_styles() +
        get_collapsible_chat_styles() +
        get_message_bubble_styles() +
        get_input_area_styles() +
        get_responsive_styles()
    )