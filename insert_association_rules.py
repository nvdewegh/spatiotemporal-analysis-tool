#!/usr/bin/env python3
"""
Script to insert Association Rules functionality into streamlit_visualization.py
"""

import sys

def insert_association_rules_functions(filepath):
    """Insert association rules functions before sequence analysis functions"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the line with "# SEQUENCE ANALYSIS FUNCTIONS"
    insert_line = None
    for i, line in enumerate(lines):
        if "# SEQUENCE ANALYSIS FUNCTIONS" in line and "# ========" in lines[i-1]:
            insert_line = i - 1  # Insert before the separator
            break
    
    if insert_line is None:
        print("Could not find insertion point")
        return False
    
    # Read the association rules functions
    ar_functions = """

# ============================================================================
# ASSOCIATION RULES INFRASTRUCTURE FUNCTIONS
# ============================================================================

def initialize_association_rules_session_state():
    \"\"\"Initialize session state variables for association rules.\"\"\"
    if 'ar_transactions' not in st.session_state:
        st.session_state.ar_transactions = None
    if 'ar_item_names' not in st.session_state:
        st.session_state.ar_item_names = None
    if 'ar_rules' not in st.session_state:
        st.session_state.ar_rules = None
    if 'ar_frequent_itemsets' not in st.session_state:
        st.session_state.ar_frequent_itemsets = None
    if 'ar_cooccurrence_matrix' not in st.session_state:
        st.session_state.ar_cooccurrence_matrix = None
    if 'ar_distance_matrix' not in st.session_state:
        st.session_state.ar_distance_matrix = None


"""
    
    # Insert functions
    lines.insert(insert_line, ar_functions)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Inserted association rules functions at line {insert_line}")
    return True

def insert_association_rules_section(filepath):
    """Insert association rules UI section before sequence analysis section"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the line with 'elif analysis_method == "Sequence Analysis":'
    insert_line = None
    for i, line in enumerate(lines):
        if 'elif analysis_method == "Sequence Analysis":' in line:
            insert_line = i
            break
    
    if insert_line is None:
        print("Could not find 'Sequence Analysis' section")
        return False
    
    # Read the association rules section from file
    with open('association_rules_section.py', 'r') as f:
        ar_section_lines = f.readlines()[2:]  # Skip first 2 comment lines
    
    # Insert section
    for line in reversed(ar_section_lines):
        lines.insert(insert_line, line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Inserted association rules section at line {insert_line}")
    return True

if __name__ == "__main__":
    filepath = "streamlit_deploy/streamlit_visualization.py"
    
    print("Step 1: Inserting helper functions...")
    if not insert_association_rules_functions(filepath):
        sys.exit(1)
    
    print("Step 2: Inserting UI section...")
    if not insert_association_rules_section(filepath):
        sys.exit(1)
    
    print("\n✅ All insertions complete!")
    print("Note: You still need to copy the functions from association_rules_functions.py")
    print("      to the ASSOCIATION RULES INFRASTRUCTURE FUNCTIONS section.")
