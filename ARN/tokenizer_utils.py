import sys
from types import ModuleType

class MockModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        sub_module = MockModule(f"{self.__name__}.{name}")
        sub_module.__path__ = []
        sub_module.__spec__ = sys.modules['os'].__spec__
        setattr(self, name, sub_module)
        sys.modules[f"{self.__name__}.{name}"] = sub_module
        return sub_module

# Create root torchvision mock
torchvision_mock = MockModule('torchvision')
torchvision_mock.__path__ = []
torchvision_mock.__spec__ = sys.modules['os'].__spec__

# Insert root mock
sys.modules['torchvision'] = torchvision_mock

# Create and register torchvision.transforms
torchvision_transforms_mock = MockModule('torchvision.transforms')
torchvision_transforms_mock.__path__ = []
torchvision_transforms_mock.__spec__ = sys.modules['os'].__spec__
sys.modules['torchvision.transforms'] = torchvision_transforms_mock
torchvision_mock.transforms = torchvision_transforms_mock

# Register InterpolationMode
class DummyInterpolationModeMeta(type):
    def __getattr__(cls, name):
        return 0

class DummyInterpolationMode(metaclass=DummyInterpolationModeMeta):
    BILINEAR = 2
    BICUBIC = 3
    NEAREST = 0
    NEAREST_EXACT = 0

torchvision_transforms_mock.InterpolationMode = DummyInterpolationMode

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from transformers import BertTokenizerFast

TOKENIZER= BertTokenizerFast.from_pretrained("bert-base-uncased")
#bio tagging (in derivation named as tag set S containing 21 possible tag slots)
TAG_TO_IDX={
    "O":0, #outside any argument
    "B-contact": 1,"I-contact":2, 
    "B-url": 3,"I-url": 4, 
    "B-app": 5, "I-app":6,
    "B-device": 7,"I-device": 8,
    "B-query": 9,"I-query": 10, 
    "B-message": 11,"I-message":12, 
    "B-action":13,"I-action":14, 
    "B-target":15,"I-target":16,
    "B-value":17,"I-value":18,
    "B-reminder":19, "I-reminder":20,
    "<PAD>": 21
}

IDX_TO_TAG= {v:k for k,v in TAG_TO_IDX.items()} #returns the tag name for a given index
# print(IDX_TO_TAG)
#tool set in derivation named as T containing 16 possible tool calls
TOOL_TO_IDX= {
    "get_current_time":0,
    "handle_smart_home": 1,
    "route_to_deepseek":2,
    "open_app_and_type": 3,
    "open_url": 4,
    "search_web": 5,
    "run_computer_command": 6,
    "analyze_screen_with_llava": 7,
    "search_and_read_web": 8,
    "read_specific_url":9,
    "write_and_run_script": 10,
    "get_system_diagnostics":11,
    "media_control":12,
    "send_whatsapp_message":13,
    "set_timer": 14,
    "teach_new_skill":15
}
IDX_TO_TOOL= {v:k for k,v in TOOL_TO_IDX.items()}
# print(IDX_TO_TOOL)

#when bert tokenizes text, its split into subwords and bio tags are assigned to full words
# therefore a helper func to align words and tags to subword tokens

def align_tags_to_tokens(words,tags,tokenizer, max_seq_len=32):
    tokenized_input= tokenizer(
        words, 
        is_split_into_words= True,
        max_length= max_seq_len,
        truncation= True,
        padding= "max_length"
    )
    #maps each subtoken index to its original word index
    word_ids= tokenized_input.word_ids()
    aligned_tag_ids=[]
    previous_word_idx= None

    for word_idx in word_ids:
        #handling special tokens and padding
        if word_idx is None:
            aligned_tag_ids.append(TAG_TO_IDX["<PAD>"])
            previous_word_idx=None
            continue
        #get original word level tag
        word_tag= tags[word_idx]
        if word_idx==previous_word_idx:
            #convert B- and I- to subword continuation
            if word_tag.startswith("B-"):
                slot_type=word_tag.split("-")[1]
                aligned_tag_ids.append(TAG_TO_IDX["I-"+slot_type])
            else:
                aligned_tag_ids.append(TAG_TO_IDX[word_tag])
        else:
            # First subtoken gets the word tag exactly
            aligned_tag_ids.append(TAG_TO_IDX[word_tag])
        previous_word_idx = word_idx
    return tokenized_input["input_ids"], tokenized_input["attention_mask"], aligned_tag_ids
