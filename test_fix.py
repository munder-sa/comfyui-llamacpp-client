#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify the fix for stop_sequences/messages type handling."""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.param_utils import clean_params, JSON_PARAMETERS


def test_clean_params_with_list_input():
    """Test that clean_params handles list inputs correctly."""
    print("=" * 60)
    print("Testing clean_params with list inputs")
    print("=" * 60)
    
    # Test 1: stop_sequences as list
    print("\n[Test 1] stop_sequences as list")
    params = {
        "stop_sequences": ["\n", "User:", "Assistant:"],
        "temperature": 0.7,
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "stop_sequences" in result, "stop_sequences should be in result"
    assert isinstance(result["stop_sequences"], list), "stop_sequences should be a list"
    assert result["stop_sequences"] == ["\n", "User:", "Assistant:"], "stop_sequences value should match"
    print("  ✓ PASSED")
    
    # Test 2: messages as list
    print("\n[Test 2] messages as list")
    params = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "temperature": 0.7,
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "messages" in result, "messages should be in result"
    assert isinstance(result["messages"], list), "messages should be a list"
    assert len(result["messages"]) == 2, "messages should have 2 items"
    print("  ✓ PASSED")
    
    # Test 3: messages as JSON string
    print("\n[Test 3] messages as JSON string")
    params = {
        "messages": '[{"role": "system", "content": "You are a helpful assistant."}]',
        "temperature": 0.7,
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "messages" in result, "messages should be in result"
    assert isinstance(result["messages"], list), "messages should be a list"
    assert len(result["messages"]) == 1, "messages should have 1 item"
    print("  ✓ PASSED")
    
    # Test 4: documents as list
    print("\n[Test 4] documents as list")
    params = {
        "documents": ["Document 1", "Document 2", "Document 3"],
        "query": "What is this about?",
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "documents" in result, "documents should be in result"
    assert isinstance(result["documents"], list), "documents should be a list"
    assert len(result["documents"]) == 3, "documents should have 3 items"
    print("  ✓ PASSED")
    
    # Test 5: logit_bias as list
    print("\n[Test 5] logit_bias as list")
    params = {
        "logit_bias": [123, 456, 789],
        "temperature": 0.7,
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "logit_bias" in result, "logit_bias should be in result"
    assert isinstance(result["logit_bias"], list), "logit_bias should be a list"
    assert result["logit_bias"] == [123, 456, 789], "logit_bias value should match"
    print("  ✓ PASSED")
    
    # Test 6: Empty list handling
    print("\n[Test 6] Empty list handling")
    params = {
        "stop_sequences": [],
        "temperature": 0.7,
    }
    result = clean_params(params)
    print(f"  Input:  {params}")
    print(f"  Output: {result}")
    assert "stop_sequences" in result, "stop_sequences should be in result"
    assert isinstance(result["stop_sequences"], list), "stop_sequences should be a list"
    assert len(result["stop_sequences"]) == 0, "stop_sequences should be empty"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_clean_params_with_list_input()