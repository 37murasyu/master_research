#!/usr/bin/env python3
"""Test script to verify adaptive LPF implementation structure."""

import ast
import sys

with open('master_research_code.py', encoding='utf-8') as f:
    tree = ast.parse(f.read())

# Extract class and function definitions
classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

passed = True

print("=" * 60)
print("ADAPTIVE LPF IMPLEMENTATION STRUCTURE CHECK")
print("=" * 60)

print("\n[Classes]")
if 'OnlineF0Estimator' in classes:
    print("  ✓ OnlineF0Estimator class defined")
else:
    print("  ✗ OnlineF0Estimator class NOT FOUND")
    passed = False

print("\n[Functions]")
if '_fc_scheduler' in functions:
    print("  ✓ _fc_scheduler function defined")
else:
    print("  ✗ _fc_scheduler function NOT FOUND")
    passed = False

if 'compute_cycle_energy_filtered' in functions:
    print("  ✓ compute_cycle_energy_filtered function defined")
else:
    print("  ✗ compute_cycle_energy_filtered function NOT FOUND")
    passed = False

if '_butter_lowpass_filtfilt' in functions:
    print("  ✓ _butter_lowpass_filtfilt function defined")
else:
    print("  ✗ _butter_lowpass_filtfilt function NOT FOUND")
    passed = False

print("\n[Settings]")
# Look for global variable assignments
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id.startswith('E_FC_'):
                    print(f"  ✓ {target.id} setting defined")

print("\n" + "=" * 60)
if passed:
    print("✓ ALL STRUCTURAL CHECKS PASSED")
    print("=" * 60)
    sys.exit(0)
else:
    print("✗ SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(1)
