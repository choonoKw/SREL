# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:27:33 2024

@author: jbk5816
"""

# def is_single_nn(state_dict):
#     state_dict_keys = model.state_dict().keys()
#     structure_type = detect_module_structure(state_dict_keys)
    
#     # Pattern to search for numerical indexing indicating a ModuleList
#     pattern = re.compile(r'\.\d+\.')
#     for key in state_dict_keys:
#         if pattern.search(key):
#             print("Multiple NN modules used (ModuleList).")
#             return 0
#     print("Single NN module reused.")
#     return 1

def is_single_nn(state_dict, module_name):
    prefixes = set()
    for key in state_dict.keys():
        if key.startswith(module_name + "."):
            # Extract what should be the index and parameter name
            potential_index = key.split(module_name + ".")[1].split('.')[0]
            try:
                # Convert the potential index to an integer to see if it's numerical
                int(potential_index)
                prefixes.add(potential_index)
            except ValueError:
                # If conversion fails, it's not a numerical index, indicating a single module case
                continue

    # If there are multiple prefixes and they form a sequence starting from 0,
    # it likely indicates a ModuleList with multiple modules
    if len(prefixes) > 1 and all(str(i) in prefixes for i in range(len(prefixes))):
        print("Multiple NN modules used (ModuleList).")
        return 0
    else:
        print("Single NN module reused.")
        return 1
    # module_keys = [key for key in state_dict.keys() if key.startswith(module_name)]
    # # Look for numerical prefixes in the parameter names
    # has_multiple_modules = any('.' in key.split(module_name + '.')[1] for key in module_keys)
    
    # if has_multiple_modules:
    #     print("Multiple NN modules used (ModuleList).")
    #     return 0
    # else:
    #     print("Single NN module reused.")
    #     return 1