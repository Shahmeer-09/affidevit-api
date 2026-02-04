"""Test smart validation with user's exact fields"""
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from affidavits.services.policy_generator_service import convert_detected_fields_to_intake_schema, TT_FIELD_RULES
import json

# Test with the exact fields shown by user
test_fields = [
    {'id': 'full_name', 'label': 'Full Name', 'type': 'text', 'required': True},
    {'id': 'age', 'label': 'Age', 'type': 'number', 'required': True},
    {'id': 'address', 'label': 'Address', 'type': 'text', 'required': True},
    {'id': 'electoral_identification_card_number', 'label': 'Electoral Identification Card Number', 'type': 'text', 'required': True},
    {'id': 'declaration_day', 'label': 'Declaration Day', 'type': 'number', 'required': True},
    {'id': 'declaration_month', 'label': 'Declaration Month', 'type': 'text', 'required': True},
]

result = convert_detected_fields_to_intake_schema(test_fields)

print('=== CONVERTED FIELDS ===')
for field in result:
    print(f"Field: {field['id']}")
    print(f"  Label: {field['label']}")
    print(f"  Type: {field['type']}")
    if field.get('_converted_from'):
        print(f"  *** CONVERTED FROM: {field['_converted_from']} ***")
    if field.get('validation'):
        print(f"  Validation: {json.dumps(field['validation'], indent=4)}")
    print()

print("=== CHECK: Electoral ID patterns ===")
print(TT_FIELD_RULES['national_id']['patterns'])
