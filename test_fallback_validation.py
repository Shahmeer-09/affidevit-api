"""
Test script to simulate OpenAI connection failure and test fallback validation
"""
import os
import django
import sys

# Setup Django  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Mock OpenAI to raise connection error
from unittest.mock import patch, MagicMock

# User's exact data with gibberish
test_data = {
    "full_name": "usman ",
    "age": "56",
    "address": "123 main street  hahahah yeah m goood  ....       ",
    "id_number": "876545678765",
    "property_description": "the house is old kldbfihdb",
    "land_ownership_details": "i own the house",
    "assessment_number": "789",
    "years_residing": "3",
    "permission_details": "the permission was given by the council ",
    "house_age": "40",
    "repair_details": "replacing windows",
    "ownership_statement": "i dont own any other property ",
    "declaration_location": "bacolet street",
    "city": "muzaffargarh",
    "day": "19th",
    "month": "december",
    "year": "2025"
}

template = """<p> <strong>REPUBLIC OF TRINIDAD AND TOBAGO:</strong></p>"""

print("=" * 80)
print("TESTING FALLBACK VALIDATION (Simulating OpenAI Connection Error)")
print("=" * 80)
print("\n📝 Test Data:")
print(f"  - address: '{test_data['address']}'")
print(f"  - property_description: '{test_data['property_description']}'")
print("\nScenario: OpenAI API is DOWN/UNAVAILABLE")
print("Expected: Fallback pattern matching should catch the gibberish")
print("\n🔍 Running validation with mocked OpenAI failure...")
print("=" * 80)

# Mock get_openai_client to raise ConnectionError
with patch('affidavits.services.ai_service.get_openai_client') as mock_client:
    mock_client.side_effect = Exception("Connection error")
    
    from affidavits.services.ai_service import validate_inputs_before_submission
    
    try:
        result = validate_inputs_before_submission(
            answers_json=test_data,
            template_html=template,
            affidavit_type_name="Test Affidavit"
        )
        
        print("\n✅ VALIDATION RESULT (Fallback Mode):")
        print(f"  all_valid: {result.get('all_valid')}")
        print(f"  fallback_validation: {result.get('fallback_validation', False)}")
        print(f"  invalid_fields: {result.get('invalid_fields')}")
        print(f"  error: {result.get('error', 'None')}")
        print(f"\n📋 Validation Notes:")
        
        if result.get('validation_notes'):
            for note in result.get('validation_notes'):
                print(f"\n  Field: {note.get('field')}")
                print(f"  Value: '{note.get('value')}'")
                print(f"  Issue: {note.get('issue')}")
                print(f"  Example: {note.get('example')}")
        else:
            print("  (no validation notes)")
        
        print("\n" + "=" * 80)
        
        if result.get('all_valid'):
            print("❌ PROBLEM: Fallback validation PASSED but should have FAILED!")
            print("   Pattern matching did NOT catch the trailing gibberish.")
        else:
            print("✅ SUCCESS: Fallback validation correctly FAILED!")
            print(f"   Caught {len(result.get('invalid_fields', {}))} invalid fields using pattern matching")
            if result.get('fallback_validation'):
                print("   ✓ Fallback mode activated (OpenAI unavailable)")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR during fallback validation: {e}")
        import traceback
        traceback.print_exc()
