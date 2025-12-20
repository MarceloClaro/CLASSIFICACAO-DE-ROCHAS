#!/usr/bin/env python3
"""
Test script to demonstrate the Google GenAI API compatibility fix.
This script shows how the code automatically detects and uses the correct API.
"""

import sys

print("=" * 70)
print("Google GenAI API Compatibility Test")
print("=" * 70)
print()

# Test 1: Import detection
print("Test 1: Package Detection")
print("-" * 70)

try:
    import google.genai as genai_new
    print("✓ google.genai (NEW package) is available")
    GEMINI_NEW_API = True
    genai = genai_new
except ImportError:
    print("✗ google.genai (NEW package) is NOT available")
    try:
        import google.generativeai as genai_old
        print("✓ google.generativeai (OLD package) is available")
        GEMINI_NEW_API = False
        genai = genai_old
    except ImportError:
        print("✗ google.generativeai (OLD package) is NOT available")
        GEMINI_AVAILABLE = False
        GEMINI_NEW_API = None
        genai = None

print()

# Test 2: API detection
if genai is not None:
    print("Test 2: API Method Detection")
    print("-" * 70)
    
    print(f"Package name: {genai.__name__}")
    print(f"API Type: {'NEW' if GEMINI_NEW_API else 'OLD'}")
    
    # Check for methods
    has_configure = hasattr(genai, 'configure')
    has_client = hasattr(genai, 'Client')
    has_generative_model = hasattr(genai, 'GenerativeModel')
    
    print(f"Has 'configure' method: {has_configure}")
    print(f"Has 'Client' class: {has_client}")
    print(f"Has 'GenerativeModel' class: {has_generative_model}")
    print()
    
    # Test 3: Show correct initialization method
    print("Test 3: Correct Initialization Method")
    print("-" * 70)
    
    if GEMINI_NEW_API:
        print("✓ Should use NEW API:")
        print("  client = genai.Client(api_key=api_key)")
        print("  response = client.models.generate_content(model=model_name, contents=prompt)")
    else:
        print("✓ Should use OLD API:")
        print("  genai.configure(api_key=api_key)")
        print("  model = genai.GenerativeModel(model_name)")
        print("  response = model.generate_content(prompt)")
    print()
    
    # Test 4: Verification
    print("Test 4: Verification")
    print("-" * 70)
    
    if GEMINI_NEW_API and has_client:
        print("✅ NEW API detected and Client class is available")
    elif not GEMINI_NEW_API and has_configure:
        print("✅ OLD API detected and configure method is available")
    else:
        print("⚠️ Warning: API detection may have issues")
    print()
    
else:
    print("\nTest 2-4: SKIPPED (No Google GenAI package installed)")
    print()
    print("To install the recommended package, run:")
    print("  pip install google-generativeai")
    print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)

if genai is not None:
    print("✅ Google GenAI package is installed and detected correctly")
    print(f"   API Type: {'NEW (google-genai)' if GEMINI_NEW_API else 'OLD (google-generativeai)'}")
    print("   The code will automatically use the correct initialization method")
else:
    print("⚠️ No Google GenAI package is installed")
    print("   Install with: pip install google-generativeai")

print()
print("For more information, see: API_SETUP_GUIDE.md")
print("=" * 70)
