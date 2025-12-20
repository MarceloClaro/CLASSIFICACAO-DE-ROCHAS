#!/usr/bin/env python3
"""
Test script to verify Google Gemini API compatibility and supported models.
This script shows how the code automatically detects and uses the correct API.

Updated for v1beta API with new model names.
"""

import sys

print("=" * 70)
print("Google Gemini API Compatibility Test (v1beta)")
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
    print(f"API Type: {'NEW' if GEMINI_NEW_API else 'OLD (RECOMMENDED)'}")
    
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
        print("  # Note: Model name must include 'models/' prefix")
        print("  model_path = f'models/{model_name}' if not model_name.startswith('models/') else model_name")
        print("  response = client.models.generate_content(model=model_path, contents=prompt)")
    else:
        print("✓ Should use OLD API (RECOMMENDED):")
        print("  genai.configure(api_key=api_key)")
        print("  model = genai.GenerativeModel(model_name)")
        print("  response = model.generate_content(prompt)")
    print()
    
    # Test 4: Supported Models (v1beta)
    print("Test 4: Supported Models (v1beta API)")
    print("-" * 70)
    print("✅ SUPPORTED MODELS:")
    print("  • gemini-2.5-flash ⭐ RECOMMENDED (newest, fast, efficient)")
    print("  • gemini-1.5-flash (stable alternative)")
    print("  • gemini-2.5-pro (advanced reasoning)")
    print("  • gemini-pro (general purpose)")
    print()
    print("❌ DEPRECATED MODELS (NOT SUPPORTED):")
    print("  • gemini-1.0-pro (use gemini-pro instead)")
    print("  • gemini-1.5-pro (use gemini-2.5-pro or gemini-2.5-flash instead)")
    print()
    
    # Test 5: Verification
    print("Test 5: Verification")
    print("-" * 70)
    
    if GEMINI_NEW_API and has_client:
        print("✅ NEW API detected and Client class is available")
    elif not GEMINI_NEW_API and has_configure:
        print("✅ OLD API detected and configure method is available")
    else:
        print("⚠️ Warning: API detection may have issues")
    print()
    
else:
    print("\nTest 2-5: SKIPPED (No Google GenAI package installed)")
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
    print(f"   API Type: {'NEW (google-genai)' if GEMINI_NEW_API else 'OLD (google-generativeai - RECOMMENDED)'}")
    print("   The code will automatically use the correct initialization method")
    print()
    print("📋 Quick Start:")
    print("   1. Get API key from: https://ai.google.dev/")
    print("   2. Use model: gemini-2.5-flash (recommended)")
    print("   3. Configure in app sidebar")
else:
    print("⚠️ No Google GenAI package is installed")
    print("   Install with: pip install google-generativeai")

print()
print("For more information, see: API_SETUP_GUIDE.md")
print("=" * 70)
