import json

def fix_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('//') or not line.strip():
            continue
            
        try:
            # Parse and re-serialize to ensure valid JSON
            obj = json.loads(line)
            fixed_line = json.dumps(obj, ensure_ascii=False)
            fixed_lines.append(fixed_line)
        except json.JSONDecodeError:
            continue
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')

# Run the fix
fix_json_file('train_requirements.jsonl', 'fixed_train_requirements.jsonl')