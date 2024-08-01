import os
import sentencepiece as spm


def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size):
    model_file = model_prefix + ".model"
    vocab_file = model_prefix + ".vocab"
    print(f"Starting training for vocab size: {vocab_size}")
    print(f"Input file: {input_file}")
    print(f"Output model file: {model_file}")
    print(f"Output vocab file: {vocab_file}")
    
    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            max_sentence_length=5000,
            hard_vocab_limit = False
            
        )
        
        if os.path.exists(model_file) and os.path.exists(vocab_file):
            print(f"model & vocab files created: {model_file}, {vocab_file}")
        else:
            print(f"Error, model or  vocab file not created for {model_prefix}")
        print("training done successfully")
    except Exception as e:
        print(f"an error occurred during training: {e}")
        raise e
        
def compression_ratio(genome,tokens,vocab):
    genome_len = len(genome)*2
    stokens=set(tokens)
    vocab_len = sum([len(v)*2+1 for e,v in enumerate(vocab) if e in stokens])-1
    sequence_len = len(' '.join([bin(i)[2:] for i in tokens]))
    return vocab_len,sequence_len,genome_len,\
    (vocab_len+sequence_len)/genome_len
