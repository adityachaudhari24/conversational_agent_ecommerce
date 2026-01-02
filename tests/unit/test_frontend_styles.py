"""Tests for frontend styles functionality."""

import pytest
from src.frontend.styles import (
    get_chat_styles,
    get_collapsible_chat_styles,
    get_message_bubble_styles,
    get_input_area_styles,
    get_responsive_styles,
    get_all_chat_styles
)


class TestFrontendStyles:
    """Test frontend styles functionality."""
    
    def test_get_chat_styles_returns_valid_css(self):
        """Test that get_chat_styles returns valid CSS string."""
        styles = get_chat_styles()
        
        assert isinstance(styles, str)
        assert len(styles) > 0
        assert "<style>" in styles
        assert "</style>" in styles
        
        # Check for key CSS classes that should be present
        assert ".chat-container" in styles
        assert ".messages-container" in styles
        assert ".chat-header" in styles
        
        # Check for requirement 5.5 - fixed width (400px)
        assert "width: 400px" in styles
        assert "adjustable height" in styles or "max-height" in styles
    
    def test_get_message_bubble_styles_returns_valid_css(self):
        """Test that message bubble styles include user/assistant distinction."""
        styles = get_message_bubble_styles()
        
        assert isinstance(styles, str)
        assert len(styles) > 0
        assert "<style>" in styles
        assert "</style>" in styles
        
        # Check for requirement 6.2 - visual distinction between user and assistant
        assert ".chat-message-content.user" in styles
        assert ".chat-message-content.assistant" in styles
        assert ".chat-avatar.user" in styles
        assert ".chat-avatar.assistant" in styles
    
    def test_get_collapsible_chat_styles_returns_valid_css(self):
        """Test that collapsible chat styles are present."""
        styles = get_collapsible_chat_styles()
        
        assert isinstance(styles, str)
        assert len(styles) > 0
        assert "<style>" in styles
        assert "</style>" in styles
        
        # Check for collapsible functionality
        assert ".chat-widget" in styles
        assert ".collapsed" in styles
        assert ".expanded" in styles
    
    def test_get_input_area_styles_returns_valid_css(self):
        """Test that input area styles are present."""
        styles = get_input_area_styles()
        
        assert isinstance(styles, str)
        assert len(styles) > 0
        assert "<style>" in styles
        assert "</style>" in styles
        
        # Check for input styling
        assert ".chat-input" in styles
        assert ".send-button" in styles
    
    def test_get_responsive_styles_returns_valid_css(self):
        """Test that responsive styles include media queries."""
        styles = get_responsive_styles()
        
        assert isinstance(styles, str)
        assert len(styles) > 0
        assert "<style>" in styles
        assert "</style>" in styles
        
        # Check for responsive behavior
        assert "@media" in styles
        assert "max-width" in styles
        assert "768px" in styles  # Common mobile breakpoint
        assert "480px" in styles  # Common small mobile breakpoint
    
    def test_get_all_chat_styles_combines_all_styles(self):
        """Test that get_all_chat_styles combines all individual style functions."""
        all_styles = get_all_chat_styles()
        
        assert isinstance(all_styles, str)
        assert len(all_styles) > 0
        
        # Should contain elements from all individual style functions
        individual_styles = [
            get_chat_styles(),
            get_collapsible_chat_styles(),
            get_message_bubble_styles(),
            get_input_area_styles(),
            get_responsive_styles()
        ]
        
        # Check that the combined styles are longer than any individual style
        for style in individual_styles:
            assert len(all_styles) > len(style)
        
        # Check that key elements from each style function are present
        assert ".chat-container" in all_styles  # from get_chat_styles
        assert ".chat-widget" in all_styles     # from get_collapsible_chat_styles
        assert ".chat-message-content" in all_styles  # from get_message_bubble_styles
        assert ".chat-input" in all_styles      # from get_input_area_styles
        assert "@media" in all_styles           # from get_responsive_styles
    
    def test_styles_contain_requirement_specific_elements(self):
        """Test that styles contain elements addressing specific requirements."""
        all_styles = get_all_chat_styles()
        
        # Requirement 5.5: Fixed width (400px) and adjustable height
        assert "width: 400px" in all_styles
        assert "max-height" in all_styles or "min-height" in all_styles
        
        # Requirement 6.2: Visual distinction between user and assistant messages
        assert ".user" in all_styles
        assert ".assistant" in all_styles
        
        # Should have different styling for user vs assistant
        user_styles = [line for line in all_styles.split('\n') if '.user' in line]
        assistant_styles = [line for line in all_styles.split('\n') if '.assistant' in line]
        
        assert len(user_styles) > 0
        assert len(assistant_styles) > 0
    
    def test_styles_are_valid_css_format(self):
        """Test that styles follow valid CSS format."""
        all_styles = get_all_chat_styles()
        
        # Basic CSS structure checks
        assert all_styles.count("<style>") == all_styles.count("</style>")
        assert "{" in all_styles
        assert "}" in all_styles
        
        # Should not contain obvious syntax errors
        assert "{{" not in all_styles  # Double braces
        assert "}}" not in all_styles  # Double braces
        
        # Should contain CSS properties
        assert ":" in all_styles
        assert ";" in all_styles