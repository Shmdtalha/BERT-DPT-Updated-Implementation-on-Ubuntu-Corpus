
with open("data/Ubuntu_Corpus_V2/train.txt", 'r',encoding='utf-8') as infile, open("data/Ubuntu_Corpus_V2/ubuntu_post_training.txt", 'w',encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        fields = line.split('\t')
        
        us_id = fields[0]
        context = fields[1]
        
        turns = context.split(' __eot__ ')

        for turn in turns:
            turn = turn.replace("__eot__", "")
            turn = turn.replace("__eou__", "")
            outfile.write(turn + "\n")
        outfile.write("\n")


import pickle

# Load responses from response.txt
def load_responses(response_file):
    response_map = {}
    with open(response_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                response_id, response_text = parts
                response_map[response_id] = response_text
    return response_map

# Define function to parse train lines
def parse_train_line(line, response_map):
    parts = line.strip().split('\t')
    if len(parts) != 4:
        raise ValueError(f"Unexpected format in line: {line}")
    
    context_raw = parts[1]
    context = context_raw.split(' __eou__ ')  # Split on the end of utterance token
    response_id1 = parts[2]
    response_id2 = parts[3]
    
    # Determine the response and label based on response IDs
    if response_id1 == 'NA':
        response = response_map.get(response_id2, response_id2)
        label = 0  # Negative response
    else:
        response = response_map.get(response_id1, response_id1)
        label = 1  # Positive response

    return {
        'utterances': context,
        'response': response,
        'label': label
    }

# Define function to parse test lines
def parse_test_line(line, response_map):
    parts = line.strip().split('\t')
    if len(parts) != 4:
        raise ValueError(f"Unexpected format in line: {line}")
    
    context_raw = parts[1]
    context = context_raw.split(' __eou__ ')  # Split on the end of utterance token
    
    valid_response_id = parts[2]
    negative_response_ids = parts[3].split('|')
    
    # Create contexts with one valid response
    valid_example = {
        'utterances': context,
        'response': response_map.get(valid_response_id, valid_response_id),
        'label': 1  # Positive response
    }
    
    # Create contexts for each negative response
    negative_contexts = []
    for neg_resp_id in negative_response_ids:
        negative_contexts.append({
            'utterances': context,
            'response': response_map.get(neg_resp_id, neg_resp_id),
            'label': 0  # Negative response
        })

    return [valid_example] + negative_contexts

# Function to create pickle file
def create_pkl_file(input_txt, output_pkl, parse_line_fn, response_map):
    contexts = []
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            contexts.extend(parse_line_fn(line, response_map))  # Add all contexts from the line

    with open(output_pkl, 'wb') as pkl_handle:
        pickle.dump(contexts, pkl_handle)

    print(f"Pickle file {output_pkl} created with {len(contexts)} contexts.")

# Load response mapping
response_map = load_responses('data/Ubuntu_Corpus_V2/responses.txt')

# Generate pickle files
create_pkl_file('data/Ubuntu_Corpus_V2/train.txt', 'data/Ubuntu_Corpus_V2/ubuntu_train.pkl', lambda line, response_map: [parse_train_line(line, response_map)], response_map)
create_pkl_file('data/Ubuntu_Corpus_V2/valid.txt', 'data/Ubuntu_Corpus_V2/ubuntu_valid.pkl', lambda line, response_map: [parse_test_line(line, response_map)], response_map)
create_pkl_file('data/Ubuntu_Corpus_V2/test.txt', 'data/Ubuntu_Corpus_V2/ubuntu_test.pkl', lambda line, response_map: [parse_test_line(line, response_map)], response_map)
