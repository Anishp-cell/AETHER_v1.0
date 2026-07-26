from dataclasses import dataclass

@dataclass(frozen=True)
class ARNconfig:
    VOCAB_SIZE= 30522 #berts word piece vocab
    EMBED_DIM_SMALL=64 #factorized embedding input dim
    HIDDEN_DIM= 128 #inner hidden dim of BERT-tiny
    FFN_DIM= 512 #FFN dim of BERT-tiny
    NUM_HEADS= 2 #attention heads of BERT-tiny
    NUM_LAYERS= 2 #num of encoder blocks
    MAX_SEQ_LEN=64
    NUM_TOOLS=16
    DROPOUT=0.1
    LR=3e-4
    BATCH_SIZE=32
    EPOCHS=30
    LAMBDA_SLOT=1.0



