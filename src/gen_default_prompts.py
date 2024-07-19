import json

defaults = {}

naive = {"sample description": ["ENTITY", ": ", "CONTENT"], "extra sample description": None,
        "task description": [" Suppose you are an expert at ", "DOMAIN", ". There are now the following ", "DOMAIN", " subcategories: ", "CATEGORY", ". What's the category of this ", "ENTITY", "? If none of the above category fits, choose the most appropriate one. Output the classification result in the format: 'Category: {classification result}'."], "extra task description": None}
defaults['naive'] = naive

simple_description = {"sample description": ["ENTITY", ": ", "CONTENT"], "extra sample description": None,
        "task description": [" Suppose you are an expert at ", "DOMAIN", ". There are now the following ", "DOMAIN", " subcategories: ", "CATEGORY", ". What's the category of this ", "ENTITY", "? If none of the above category fits, choose the most appropriate one. Output the classification result in the format: 'Category: {classification result}'."], "extra task description": ["DESCRIPTION"]}
defaults['simp_descrip'] = simple_description

with open("../prompts/default_prompts.json", "w") as json_file:
    json.dump(defaults, json_file)
