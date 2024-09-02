
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

