#!/usr/bin/env python3
"""
Example configurations for simple_test.py

Copy any of these configurations to the TEST_FLAGS section in simple_test.py
to run specific combinations of tests.
"""

# Example 1: Run only basic retrieval test
BASIC_ONLY = {
    "basic_retrieval": True,
    "filtered_retrieval": False,
    "context_compression": False,
    "query_rewriting": False,
}

# Example 2: Test compression and rewriting features
ADVANCED_FEATURES = {
    "basic_retrieval": False,
    "filtered_retrieval": False,
    "context_compression": True,
    "query_rewriting": True,
}

# Example 3: Test all retrieval methods
RETRIEVAL_METHODS = {
    "basic_retrieval": True,
    "filtered_retrieval": True,
    "context_compression": False,
    "query_rewriting": False,
}

# Example 4: Test processing pipeline
PROCESSING_PIPELINE = {
    "basic_retrieval": False,
    "filtered_retrieval": False,
    "context_compression": True,
    "query_rewriting": True,
}

# Example 5: Run all tests (default)
ALL_TESTS = {
    "basic_retrieval": True,
    "filtered_retrieval": True,
    "context_compression": True,
    "query_rewriting": True,
}

print("Copy one of these configurations to simple_test.py:")
print("\n# Example 1: Basic retrieval only")
print("TEST_FLAGS =", BASIC_ONLY)
print("\n# Example 2: Advanced features only")
print("TEST_FLAGS =", ADVANCED_FEATURES)
print("\n# Example 3: Retrieval methods comparison")
print("TEST_FLAGS =", RETRIEVAL_METHODS)
print("\n# Example 4: Processing pipeline")
print("TEST_FLAGS =", PROCESSING_PIPELINE)
print("\n# Example 5: All tests")
print("TEST_FLAGS =", ALL_TESTS)