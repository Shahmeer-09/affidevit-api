"""
Test script to check validation function with the user's exact data
"""
import os
import django
import sys

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from affidavits.services.ai_service import validate_inputs_before_submission

# User's exact data
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

template = """<p> <strong>REPUBLIC OF TRINIDAD AND TOBAGO:</strong></p>
<p><strong> IN THE MATTER OF THE STATUTORY DECLARATION ACT</strong></p>
<p><strong> </strong></p>
<p><strong> CHAPTER 7: No. 04</strong></p>
<p> I, <strong>{{full_name}}</strong>, age {{age}} years, of {{address}}, holding Trinidad and Tobago Electoral Identification Card No. <strong>{{id_number}}</strong> do solemnly and sincerely declare as follows: -</p>
<ol><li>That I am the declarant herein.</li>
<li>That I am in the process of making modifications and improvements to {{property_description}} on Land for which {{land_ownership_details}} and assessed as <strong>{{assessment_number}}</strong>.</li>
<li>That I have been residing on the said Land undisturbed for over {{years_residing}} years.</li>
<li>That {{permission_details}} has permitted me to make modifications and improvements to the said dwelling house on the said Land and also apply for funding to aid in effecting same.</li>
<li>That the said dwelling house is approximately {{house_age}} years old and in need of repairs and improvements which include {{repair_details}}.</li>
<li>That I am applying for Funding to aid in the effecting the modifications and improvements to the said dwelling house on the said Land.</li>
<li>That there is no dispute in the ownership of the Land.</li>
<li>{{ownership_statement}}</li>
</ol>"""

print("=" * 80)
print("TESTING VALIDATION WITH USER DATA")
print("=" * 80)
print("\n📝 Test Data:")
print(f"  - address: '{test_data['address']}'")
print(f"  - property_description: '{test_data['property_description']}'")
print("\nExpected: INVALID (both have trailing gibberish)")
print("\n🔍 Running validation...")
print("=" * 80)

try:
    result = validate_inputs_before_submission(
        answers_json=test_data,
        template_html=template,
        affidavit_type_name="Test Affidavit"
    )
    
    print("\n✅ VALIDATION RESULT:")
    print(f"  all_valid: {result.get('all_valid')}")
    print(f"  invalid_fields: {result.get('invalid_fields')}")
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
        print("❌ PROBLEM: Validation PASSED but should have FAILED!")
        print("   The validator did NOT catch the trailing gibberish.")
    else:
        print("✅ SUCCESS: Validation correctly FAILED!")
        print(f"   Caught {len(result.get('invalid_fields', {}))} invalid fields")
    
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR during validation: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
