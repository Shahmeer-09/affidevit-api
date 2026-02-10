"""
Test script for validation rules system
"""
import os
import django
import sys

# Setup Django  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from affidavits.services.ai_service import apply_validation_rules, generate_rule_summary_for_ai

print("=" * 80)
print("TESTING VALIDATION RULES ENGINE")
print("=" * 80)

# Test 1: Comparison rule (age >= residence_duration)
print("\n### Test 1: Comparison Rule (age >= residence_duration)")
test_rules_1 = [
    {
        "id": "rule-1",
        "type": "comparison",
        "primary_field": "age",
        "secondary_field": "residence_duration",
        "operator": "gte",
        "compare_as": "number",
        "target_field": "residence_duration",
        "message": "You cannot have lived somewhere longer than your age."
    }
]

test_data_1_valid = {
    "age": "30",
    "residence_duration": "for over 3 years"
}

test_data_1_invalid = {
    "age": "30",
    "residence_duration": "50 years"
}

print("\nTest 1a: Valid data (age=30, residence=3)")
invalid_fields, notes = apply_validation_rules(test_data_1_valid, test_rules_1)
print(f"  Invalid fields: {invalid_fields}")
print(f"  Expected: {{}}")
print(f"  ✓ PASS" if len(invalid_fields) == 0 else f"  ✗ FAIL")

print("\nTest 1b: Invalid data (age=30, residence=50)")
invalid_fields, notes = apply_validation_rules(test_data_1_invalid, test_rules_1)
print(f"  Invalid fields: {invalid_fields}")
print(f"  Expected: {{'residence_duration': '...'}}")
print(f"  ✓ PASS" if 'residence_duration' in invalid_fields else f"  ✗ FAIL")

# Test 2: Required_if rule
print("\n### Test 2: Required_if Rule")
test_rules_2 = [
    {
        "id": "rule-2",
        "type": "required_if",
        "condition_field": "co_signer_required",
        "condition_value": "yes",
        "required_field": "co_signer_name",
        "primary_field": "co_signer_name",
        "target_field": "co_signer_name",
        "message": "Co-signer name is required when co-signer is needed."
    }
]

test_data_2_valid = {
    "co_signer_required": "yes",
    "co_signer_name": "John Doe"
}

test_data_2_invalid = {
    "co_signer_required": "yes",
    "co_signer_name": ""
}

test_data_2_not_required = {
    "co_signer_required": "no",
    "co_signer_name": ""
}

print("\nTest 2a: Valid (condition met, field filled)")
invalid_fields, notes = apply_validation_rules(test_data_2_valid, test_rules_2)
print(f"  Invalid fields: {invalid_fields}")
print(f"  ✓ PASS" if len(invalid_fields) == 0 else f"  ✗ FAIL")

print("\nTest 2b: Invalid (condition met, field empty)")
invalid_fields, notes = apply_validation_rules(test_data_2_invalid, test_rules_2)
print(f"  Invalid fields: {invalid_fields}")
print(f"  ✓ PASS" if 'co_signer_name' in invalid_fields else f"  ✗ FAIL")

print("\nTest 2c: Valid (condition not met, field empty)")
invalid_fields, notes = apply_validation_rules(test_data_2_not_required, test_rules_2)
print(f"  Invalid fields: {invalid_fields}")
print(f"  ✓ PASS" if len(invalid_fields) == 0 else f"  ✗ FAIL")

# Test 3: Text phrase extraction
print("\n### Test 3: Text Phrase Extraction")
test_rules_3 = [
    {
        "id": "rule-3",
        "type": "comparison",
        "primary_field": "house_age",
        "secondary_field": "residence_duration",
        "operator": "gte",
        "compare_as": "number",
        "target_field": "residence_duration",
        "message": "You cannot have lived in a house longer than it has existed."
    }
]

test_data_3_valid = {
    "house_age": "over twenty-five (25) years old",
    "residence_duration": "since 2010"  # Should extract as ~14 years
}

print("\nTest 3: Text phrases (house_age='25 years', residence='since 2010')")
invalid_fields, notes = apply_validation_rules(test_data_3_valid, test_rules_3)
print(f"  Invalid fields: {invalid_fields}")
print(f"  Expected: {{}}")
print(f"  ✓ PASS" if len(invalid_fields) == 0 else f"  ✗ FAIL")

# Test 4: AI Rule Summary Generation
print("\n### Test 4: AI Rule Summary Generation")
all_rules = test_rules_1 + test_rules_2
summary = generate_rule_summary_for_ai(all_rules)
print("\nGenerated summary for AI:")
print(summary)
assert summary.strip() != "", "Summary should not be empty"
print("  ✓ Summary generated successfully")

# =====================================================================
# Test 5: Chained Comparisons (AND/OR) + NOT
# =====================================================================
print("\n### Test 5: Chained Comparison Rules (AND/OR/NOT)")

chained_rules = [
    {
        "type": "comparison",
        "comparisons": [
            {"left_field": "a", "operator": "lt", "right_field": "b", "compare_as": "number"},
            {"join_with": "AND", "left_field": "b", "operator": "gt", "right_field": "c", "compare_as": "number"},
        ],
        "target_field": "b",
        "message": "Chain must be A < B AND B > C",
    }
]

# Valid chain: 1 < 3 AND 3 > 2
invalid_fields, _ = apply_validation_rules({"a": "1", "b": "3", "c": "2"}, chained_rules)
print(f"  Valid chain invalid_fields: {invalid_fields}")
assert invalid_fields == {}, "Chained AND rule should pass"
print("  ✓ PASS")

# Invalid chain: 1 < 3 AND 3 > 5 (fails second)
invalid_fields, _ = apply_validation_rules({"a": "1", "b": "3", "c": "5"}, chained_rules)
print(f"  Invalid chain invalid_fields: {invalid_fields}")
assert "b" in invalid_fields, "Chained AND rule should fail"
print("  ✓ PASS")

or_rules = [
    {
        "type": "comparison",
        "comparisons": [
            {"left_field": "a", "operator": "lt", "right_field": "b", "compare_as": "number"},
            {"join_with": "OR", "left_field": "b", "operator": "gt", "right_field": "c", "compare_as": "number"},
        ],
        "target_field": "b",
        "message": "Chain must be A < B OR B > C",
    }
]

# OR should pass if either clause true
invalid_fields, _ = apply_validation_rules({"a": "10", "b": "3", "c": "2"}, or_rules)
print(f"  OR chain invalid_fields: {invalid_fields}")
assert invalid_fields == {}, "Chained OR rule should pass when one clause true"
print("  ✓ PASS")

not_rules = [
    {
        "type": "comparison",
        "negate": True,
        "comparisons": [
            {"left_field": "a", "operator": "lt", "right_field": "b", "compare_as": "number"},
        ],
        "target_field": "a",
        "message": "NOT (A < B) must be true",
    }
]

invalid_fields, _ = apply_validation_rules({"a": "1", "b": "2"}, not_rules)
print(f"  NOT rule invalid_fields: {invalid_fields}")
assert "a" in invalid_fields, "NOT rule should fail when underlying condition is true"
print("  ✓ PASS")

# =====================================================================
# Test 6: Disallow Contains Rule
# =====================================================================
print("\n### Test 6: Disallow Contains Rule")

contains_rules = [
    {
        "type": "disallow_contains",
        "field": "address",
        "pattern": "asdf",
        "mode": "contains",
        "case_sensitive": False,
        "target_field": "address",
        "message": "Address contains disallowed gibberish",
    }
]

invalid_fields, _ = apply_validation_rules({"address": "15 Queen Street asdf"}, contains_rules)
print(f"  Disallow contains invalid_fields: {invalid_fields}")
assert "address" in invalid_fields, "Disallow contains should flag when substring is present"
print("  ✓ PASS")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
