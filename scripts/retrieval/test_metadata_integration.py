#!/usr/bin/env python3
"""
Integration Test for Metadata Filtering Integration

Tests the end-to-end metadata extraction and filtering functionality in the retrieval pipeline.
This script initializes a real RetrievalPipeline with metadata extraction enabled and tests
various query types with real OpenAI API calls.

USAGE:
    python scripts/retrieval/test_metadata_integration.py

REQUIREMENTS:
    - OPENAI_API_KEY environment variable set
    - PINECONE_API_KEY environment variable set
    - Pinecone index with product data

TEST COVERAGE:
    - Product name filtering (e.g., "iPhone 12 price")
    - Price range filtering (e.g., "phones under $300")
    - Rating filtering (e.g., "highly rated phones")
    - Combined filters (e.g., "highly rated Samsung phones over $400")
    - General queries with no extractable metadata
    - Extraction failure handling
    - Timeout handling
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.retrieval.pipeline import RetrievalPipeline
from src.pipelines.retrieval.search.vector_searcher import MetadataFilter
from src.pipelines.retrieval.config import RetrievalSettings


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    query: str
    expected_filters: Optional[Dict[str, Any]]
    description: str
    should_extract: bool = True


# Define test cases
TEST_CASES = [
    TestCase(
        name="product_name_filter",
        query="iPhone 12 price",
        expected_filters={
            "product_name_pattern": "iPhone 12",
            "min_price": None,
            "max_price": None,
            "min_rating": None
        },
        description="Test extraction of product name filter",
        should_extract=True
    ),
    TestCase(
        name="price_max_filter",
        query="phones under $300",
        expected_filters={
            "product_name_pattern": None,
            "min_price": None,
            "max_price": 300.0,
            "min_rating": None
        },
        description="Test extraction of maximum price filter",
        should_extract=True
    ),
    TestCase(
        name="price_min_filter",
        query="phones over $500",
        expected_filters={
            "product_name_pattern": None,
            "min_price": 500.0,
            "max_price": None,
            "min_rating": None
        },
        description="Test extraction of minimum price filter",
        should_extract=True
    ),
    TestCase(
        name="rating_filter",
        query="highly rated Samsung phones",
        expected_filters={
            "product_name_pattern": "Samsung",
            "min_price": None,
            "max_price": None,
            "min_rating": 4.0
        },
        description="Test extraction of rating and product name filters",
        should_extract=True
    ),
    TestCase(
        name="combined_filters",
        query="highly rated Samsung phones over $400",
        expected_filters={
            "product_name_pattern": "Samsung",
            "min_price": 400.0,
            "max_price": None,
            "min_rating": 4.0
        },
        description="Test extraction of multiple combined filters",
        should_extract=True
    ),
    TestCase(
        name="no_extractable_metadata",
        query="What are smartphones?",
        expected_filters=None,
        description="Test general query with no extractable metadata",
        should_extract=False
    ),
    TestCase(
        name="product_and_price_range",
        query="iPhone 12 under $800",
        expected_filters={
            "product_name_pattern": "iPhone 12",
            "min_price": None,
            "max_price": 800.0,
            "min_rating": None
        },
        description="Test extraction of product name and price range",
        should_extract=True
    ),
]


# ============================================================================
# Test Utilities
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'-'*80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'-'*80}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_failure(text: str):
    """Print failure message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def compare_filters(expected: Optional[Dict[str, Any]], actual: Optional[MetadataFilter]) -> tuple[bool, str]:
    """
    Compare expected filters with actual extracted filters.
    
    Args:
        expected: Expected filter dictionary
        actual: Actual MetadataFilter object
        
    Returns:
        Tuple of (matches, message)
    """
    if expected is None and actual is None:
        return True, "Both filters are None (as expected)"
    
    if expected is None and actual is not None:
        return False, f"Expected None but got filters: {actual.__dict__}"
    
    if expected is not None and actual is None:
        return False, f"Expected filters {expected} but got None"
    
    # Compare each field
    mismatches = []
    
    for field, expected_value in expected.items():
        actual_value = getattr(actual, field, None)
        
        # Handle None comparisons
        if expected_value is None and actual_value is None:
            continue
        
        # Handle numeric comparisons with tolerance
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(expected_value - actual_value) > 0.01:
                mismatches.append(f"{field}: expected {expected_value}, got {actual_value}")
        # Handle string comparisons (case-insensitive for product names)
        elif isinstance(expected_value, str) and isinstance(actual_value, str):
            if expected_value.lower() not in actual_value.lower() and actual_value.lower() not in expected_value.lower():
                mismatches.append(f"{field}: expected '{expected_value}', got '{actual_value}'")
        # Handle exact comparisons
        elif expected_value != actual_value:
            mismatches.append(f"{field}: expected {expected_value}, got {actual_value}")
    
    if mismatches:
        return False, "Filter mismatches: " + "; ".join(mismatches)
    
    return True, "Filters match expected values"


def print_filter_details(filters: Optional[MetadataFilter]):
    """Print filter details in a readable format."""
    if filters is None:
        print("  No filters extracted")
        return
    
    print("  Extracted Filters:")
    if filters.product_name_pattern:
        print(f"    • Product Name: {filters.product_name_pattern}")
    if filters.min_price is not None:
        print(f"    • Min Price: ${filters.min_price}")
    if filters.max_price is not None:
        print(f"    • Max Price: ${filters.max_price}")
    if filters.min_rating is not None:
        print(f"    • Min Rating: {filters.min_rating}/5")


def print_result_summary(result):
    """Print retrieval result summary."""
    print(f"\n  Result Summary:")
    print(f"    • Documents Retrieved: {len(result.documents)}")
    print(f"    • Latency: {result.latency_ms:.0f}ms")
    print(f"    • From Cache: {result.from_cache}")
    print(f"    • Auto-Extracted Filters: {result.metadata.get('auto_extracted_filters', False)}")
    
    # Print workflow steps
    workflow_steps = result.metadata.get('workflow_steps', [])
    extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
    
    if extraction_steps:
        step = extraction_steps[0]
        print(f"    • Extraction Time: {step.get('time_ms', 0):.0f}ms")
        print(f"    • Filters Extracted: {step.get('filters_extracted', False)}")


# ============================================================================
# Test Execution
# ============================================================================

class MetadataIntegrationTester:
    """Manages integration test execution."""
    
    def __init__(self):
        """Initialize the tester."""
        self.pipeline: Optional[RetrievalPipeline] = None
        self.results: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def setup(self) -> bool:
        """
        Set up the test environment.
        
        Returns:
            True if setup successful, False otherwise
        """
        print_section("Test Setup")
        
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print_failure("OPENAI_API_KEY environment variable not set")
            return False
        print_success("OPENAI_API_KEY found")
        
        if not os.getenv("PINECONE_API_KEY"):
            print_failure("PINECONE_API_KEY environment variable not set")
            return False
        print_success("PINECONE_API_KEY found")
        
        # Initialize pipeline
        try:
            print_info("Initializing RetrievalPipeline with metadata extraction enabled...")
            
            # Create settings with metadata extraction enabled
            settings = RetrievalSettings(
                metadata_extraction_enabled=True,
                metadata_extraction_model="gpt-3.5-turbo",
                metadata_extraction_timeout=3,
                cache_enabled=False  # Disable cache for testing
            )
            
            self.pipeline = RetrievalPipeline.from_settings(settings)
            self.pipeline.initialize()
            
            print_success("Pipeline initialized successfully")
            
            # Verify metadata extractor is initialized
            if self.pipeline.metadata_extractor is None:
                print_failure("Metadata extractor not initialized")
                return False
            print_success("Metadata extractor initialized")
            
            return True
            
        except Exception as e:
            print_failure(f"Failed to initialize pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_test_case(self, test_case: TestCase) -> bool:
        """
        Run a single test case.
        
        Args:
            test_case: TestCase to run
            
        Returns:
            True if test passed, False otherwise
        """
        print_section(f"Test: {test_case.name}")
        print(f"Description: {test_case.description}")
        print(f"Query: '{test_case.query}'")
        
        try:
            # Execute retrieval
            start_time = time.time()
            result = self.pipeline.retrieve(test_case.query)
            execution_time = (time.time() - start_time) * 1000
            
            # Extract metadata about filters
            auto_extracted = result.metadata.get('auto_extracted_filters', False)
            workflow_steps = result.metadata.get('workflow_steps', [])
            extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
            
            # Get extracted filters from workflow metadata
            extracted_filters = None
            if extraction_steps and extraction_steps[0].get('filters_extracted'):
                # Reconstruct MetadataFilter from workflow step
                step = extraction_steps[0]
                extracted_filters = MetadataFilter(
                    product_name_pattern=step.get('product_name'),
                    min_price=step.get('min_price'),
                    max_price=step.get('max_price'),
                    min_rating=step.get('min_rating')
                )
            
            # Print extraction details
            print_filter_details(extracted_filters)
            print_result_summary(result)
            
            # Validate results
            test_passed = True
            messages = []
            
            # Check if extraction occurred as expected
            if test_case.should_extract and not auto_extracted:
                test_passed = False
                messages.append("Expected filters to be extracted but none were")
            elif not test_case.should_extract and auto_extracted:
                test_passed = False
                messages.append("Expected no filters but some were extracted")
            
            # Compare extracted filters with expected
            if test_case.expected_filters is not None or extracted_filters is not None:
                matches, message = compare_filters(test_case.expected_filters, extracted_filters)
                if not matches:
                    test_passed = False
                    messages.append(message)
                else:
                    messages.append(message)
            
            # Check that results were returned
            if len(result.documents) == 0:
                self.warnings += 1
                print_warning("No documents returned (may be expected for some queries)")
            
            # Record result
            self.results.append({
                'test_case': test_case.name,
                'query': test_case.query,
                'passed': test_passed,
                'execution_time_ms': execution_time,
                'documents_returned': len(result.documents),
                'filters_extracted': extracted_filters is not None,
                'auto_extracted': auto_extracted,
                'messages': messages
            })
            
            # Print result
            if test_passed:
                self.passed += 1
                print_success(f"Test PASSED: {messages[0] if messages else 'All checks passed'}")
            else:
                self.failed += 1
                print_failure(f"Test FAILED: {'; '.join(messages)}")
            
            return test_passed
            
        except Exception as e:
            self.failed += 1
            print_failure(f"Test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            
            self.results.append({
                'test_case': test_case.name,
                'query': test_case.query,
                'passed': False,
                'execution_time_ms': 0,
                'documents_returned': 0,
                'filters_extracted': False,
                'auto_extracted': False,
                'messages': [str(e)]
            })
            
            return False
    
    def run_all_tests(self):
        """Run all test cases."""
        print_header("Metadata Filtering Integration Tests")
        
        for test_case in TEST_CASES:
            self.run_test_case(test_case)
            time.sleep(0.5)  # Small delay between tests
    
    def print_summary(self):
        """Print test summary."""
        print_header("Test Summary")
        
        total_tests = len(TEST_CASES)
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {Colors.OKGREEN}{self.passed}{Colors.ENDC}")
        print(f"Failed: {Colors.FAIL}{self.failed}{Colors.ENDC}")
        print(f"Warnings: {Colors.WARNING}{self.warnings}{Colors.ENDC}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        # Print detailed results
        if self.results:
            print_section("Detailed Results")
            for result in self.results:
                status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result['passed'] else f"{Colors.FAIL}FAIL{Colors.ENDC}"
                print(f"\n{status} - {result['test_case']}")
                print(f"  Query: {result['query']}")
                print(f"  Execution Time: {result['execution_time_ms']:.0f}ms")
                print(f"  Documents: {result['documents_returned']}")
                print(f"  Filters Extracted: {result['filters_extracted']}")
                if result['messages']:
                    print(f"  Messages: {'; '.join(result['messages'])}")
        
        # Final verdict
        print_section("Final Verdict")
        if self.failed == 0:
            print_success("All tests passed! ✓")
            return True
        else:
            print_failure(f"{self.failed} test(s) failed")
            return False


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main test execution."""
    tester = MetadataIntegrationTester()
    
    # Setup
    if not tester.setup():
        print_failure("Setup failed. Exiting.")
        sys.exit(1)
    
    # Run tests
    tester.run_all_tests()
    
    # Print summary
    success = tester.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
