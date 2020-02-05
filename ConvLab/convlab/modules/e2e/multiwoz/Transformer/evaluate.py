import random
import os
import torch
import copy
import json
import torch.nn.functional as F
import time
import numpy as np
from convlab.modules.e2e.multiwoz.Transformer.train import SPECIAL_TOKENS_V1, SPECIAL_TOKENS_V4,\
    build_input_from_segments_v1, build_input_from_segments_v2, act_name, slot_name
from convlab.modules.util.multiwoz.dbquery import query
from convlab.modules.e2e.multiwoz.Transformer.pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer
from convlab.modules.e2e.multiwoz.Transformer.util import download_model_from_googledrive
from spacy.symbols import ORTH
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm
DEFAULT_CUDA_DEVICE=-1
DEFAULT_DIRECTORY = "models"

SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people',
         'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure', 'train-arriveby',
         'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange', 'restaurant-area',
         'attraction-name', 'restaurant-name', 'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure',
         'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']

class Transformer():

    def __init__(self,
                 model='gpt2_v1',
                 model_checkpoint='./models/v1',
                 max_history=15,
                 device='cuda',
                 no_sample=True,
                 max_length=40,
                 min_length=1,
                 seed=42,
                 temperature=0.9,
                 top_k=0,
                 top_p=0.8):

        if not os.path.isdir("./models"):
            download_model_from_googledrive(file_id='1ZiFYKgQRmvhk3GmgevA1B_tyumJn928M',
                                            dest_path='./models/models.zip')
        self.model_checkpoint = model_checkpoint
        self.max_history = max_history
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_sample = no_sample
        self.device = device
        self.seed = seed
        self.domains = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'police', 'hospital']
        self.cs_mapping = {'taxi': ['leaveat', 'destination', 'departure', 'arriveby'],
                           'restaurant': ['book-time','book-day','book-people','food', 'pricerange', 'name', 'area'],
                           'hospital': ['department', 'phone'],
                           'hotel': ['book-stay', 'book-day', 'book-people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type'],
                           'attraction': ['type', 'name', 'area'],
                           'train': ['book-people','leaveat', 'destination', 'day', 'arriveby', 'departure'],
                           'taxi': [],
                           'police': []}
        dia_act = open('./data/multiwoz/dialog_act_slot.txt', 'r')
        f = dia_act.read().split('\n')
        self.dia_act_dict = {}
        key = ""
        for i, c in enumerate(f):
            if i == 0:  continue  # User Dialog Act case
            t = c.split('\t')
            if len(t) == 1:
                key = t[0].lower()
                self.dia_act_dict[key] = []
            else:
                self.dia_act_dict[key].append(t[-1].strip().lower())
        random.seed(self.seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.cur_dom = ''
        self.prev_dom = ''
        self.prev_cs = []

        # what [JG] added
        self.model_name = model

        tokenizer_class = GPT2Tokenizer
        model_class = GPT2DoubleHeadsModel

        self.model = model_class.from_pretrained(self.model_checkpoint)

        if 'v1' in self.model_name:
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_V1
        else:
            self.SPECIAL_TOKENS = SPECIAL_TOKENS_V4
        if "gpt2" in self.model_name:
            self.tokenizer = tokenizer_class.from_pretrained(model_checkpoint, unk_token='<|unkwn|>')
            SPECIAL_TOKENS_DICT = {}
            for st in self.SPECIAL_TOKENS:
                SPECIAL_TOKENS_DICT[st] = st
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
            self.model.resize_token_embeddings(len(self.tokenizer))

        else:
            self.tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
            self.tokenizer.set_special_tokens(self.SPECIAL_TOKENS)
            self.model.set_num_special_tokens(len(self.SPECIAL_TOKENS))
            for s in self.SPECIAL_TOKENS:
                self.tokenizer.nlp.tokenizer.add_special_case(s, [{ORTH: s}])
        self.model.to(self.device)
        self.model.eval()
        self.count = 0
        self.reset()


    def sample_sequence_v4(self, history, current_output=None):

        special_tokens_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)

        dptok = [special_tokens_id[5]]
        sys = [special_tokens_id[3]]
        eos = [special_tokens_id[1]]

        if current_output is None:
            current_output = []

        cs_dict = {}

        i = 0
        cs_count = 0

        dp = []
        cs = []

        cs_done = 0
        dp_done = 0

        while i < self.max_length:

            instance, sequence = build_input_from_segments_v1(history, current_output, self.tokenizer, dp=dp, cs=cs,
                                                              with_eos=False, mode='infer')

            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            logits, attentions = self.model(input_ids, token_type_ids=token_type_ids)

            if "gpt2" in self.model_name:

                logits = logits[0]

            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)

            if not dp_done:
                prev = torch.topk(probs, 1)[1]
            else:
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

            if i < self.min_length and prev.item() in eos:
                b = 0
                while prev.item() in eos:
                    if b == 3:
                        break
                    prev = torch.multinomial(probs, num_samples=1)
                    b += 1


            if prev.item() in dptok:

                if cs_count == 0:

                    cs_text = self.decode(cs).strip()
                    if self.cur_dom != cs_text.split(' ')[0][1:-1] and cs_text.split(' ')[0][1:-1] in self.domains:
                        self.cur_dom = cs_text.split(' ')[0][1:-1]

                    keys = self.cs_mapping[self.cur_dom] if self.cur_dom else []

                    if keys != []:

                        prev_key = (0, '')
                        cs_tok = cs_text.split(' ')
                        for j, tok in enumerate(cs_tok):
                            if tok[1:-1] in keys:
                                if prev_key[1] != '':
                                    cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1: j])
                                prev_key = (j, tok[1:-1])
                            if j == len(cs_tok) - 1:
                                cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1:])

                    break

            if not cs_done:
                cs.append(prev.item())

            i += 1

        self.prev_dom = self.cur_dom
        return self.cur_dom, cs_dict


    def decode(self, ids, skip_special_tokens=False):

        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        if not "gpt2" in self.model_name:  # gpt
            return text

        def list_duplicates_of(seq, item):
            start_at = -1
            locs = []
            while True:
                try:
                    loc = seq.index(item, start_at + 1)
                except ValueError:
                    break
                else:
                    locs.append(loc)
                    start_at = loc
            return locs

        for st in self.SPECIAL_TOKENS:
            indices = list_duplicates_of(text, st)
            if indices:
                indices.sort()
                index_count = 0
                for index in indices:
                    real_index = index + index_count
                    text = text[:real_index] + ' ' + text[real_index:]
                    text = text[:real_index + len(st) + 1] + ' ' + text[real_index + len(st) + 1:]
                    index_count += 2
        text = text.replace('  ', ' ')
        return text


    def convert_kb(self, kb_results):

        new_kb = {}
        for key in kb_results:

            value = kb_results[key]
            if key == 'arriveBy':
                key = 'arrive'
            elif key == 'leaveAt':
                key = 'leave'
            elif key == 'trainID':
                key = 'id'
            elif key == 'Ref':
                key = 'ref'
            elif key == 'address':
                key = 'addr'
            elif key == 'duration':
                key = 'time'
            elif key == 'postcode':
                key = 'post'
            new_kb[key] = value

        return new_kb

    def top_filtering(self, logits, threshold=-float('Inf'), filter_value=-float('Inf')):

        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        self.top_k = min(self.top_k, logits.size(-1))
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def init_session(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.history = []
        self.prev_cs = []
        self.cur_dom = ''
        self.prev_dom = ''

    def predict(self, history):

        self.t += 1
        self.history = [self.tokenizer.encode(x.lower()) for x in history[-(2 * self.max_history + 1):]]
        with torch.no_grad():
            if 'v1' in self.model_name:
                cs_dict = self.sample_sequence_v1(self.history)
            else:
                dom, cs_dict = self.sample_sequence_v4(self.history)

        return dom, cs_dict


def fix_general_label_error(labels, type, slots):
    label_dict = dict([(l[0], l[1]) for l in labels]) if type else dict(
        [(l["slots"][0][0], l["slots"][0][1]) for l in labels])
    # if type:
    #     label_dict = dict([(l[0], l[1]) for l in labels])
    # else:
    #     d = []
    #     for l in labels:
    #         if not 'book' in l["slots"][0][0]:
    #             k = l["slots"][0][1]
    #             if "2 two" in l["slots"][0][1]:
    #                 k = k.replace("2 two", "two two")
    #             d.append((l["slots"][0][0], k))
    #         else:
    #             continue
    #
    #     label_dict = dict(d)


    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house", "guesthouses": "guest house", "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports", "mutliple sports": "multiple sports", "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall", "pool": "swimming pool", "night club": "nightclub", "mus": "museum",
        "ol": "architecture",
        "colleges": "college", "coll": "college", "architectural": "architecture", "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre", "center of town": "centre", "near city center": "centre", "in the north": "north",
        "cen": "centre", "east side": "east",
        "east area": "east", "west part of town": "west", "ce": "centre", "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre", "the south": "south", "scentre": "centre", "town centre": "centre",
        "in town": "centre", "north part of town": "north",
        "centre of town": "centre", "cb30aq": "none",
        # price
        "mode": "moderate", "moderate -ly": "moderate", "mo": "moderate",
        # day
        "next friday": "friday", "monda": "monday",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4", "4 stars": "4", "0 star rarting": "none",
        # others
        "y": "yes", "any": "dontcare", "n": "no", "does not care": "dontcare", "not men": "none", "not": "none",
        "not mentioned": "none",
        '': "none", "not mendtioned": "none", "3 .": "3", "does not": "no", "fun": "none", "art": "none",
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast",
                                                             "centre", "venetian", "intern", "a cheap -er hotel"] or \
                    slot == "hotel-internet" and label_dict[slot] == "4" or \
                    slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                    slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery",
                                                                       "science", "m"] or \
                    "area" in slot and label_dict[slot] in ["moderate"] or \
                    "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4",
                                                               "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                    slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum",
                                                                       "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict


def convert_cs(cs_dict):
    ret = []
    for d in cs_dict.keys():
        for k in cs_dict[d].keys():
            if not cs_dict[d][k] in ['<nm>', '<nm> ', '']:
                if 'book-' in k:
                    ret.append(d + '-book ' + k.split('-')[-1]+'-'+cs_dict[d][k])
                else:
                    ret.append(d + '-' + k + '-' + cs_dict[d][k])

    return ret

def combine_dict(prev, dom, cs_dict):
    if prev == {}:
        return {dom: cs_dict}
    if '' in cs_dict.keys():
        del(cs_dict[''])
    if cs_dict == {}:
        return prev
    new = copy.deepcopy(prev)
    if dom in prev.keys():
        for k in prev[dom]:
            if prev[dom][k] == '<nm>' and k in cs_dict.keys():
                new[dom][k] = cs_dict[k]
            else:
                if k in cs_dict.keys():
                    if prev[dom][k] != cs_dict[k]:
                        new[dom][k] = cs_dict[k]
    else:
        new[dom] = cs_dict

    return new

def run():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt2-v4', help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint", type=str, default='./models/v4_1', help="Path, url or short name of the model")
    args = parser.parse_args()

    model = Transformer(model=args.model, model_checkpoint=args.model_checkpoint)
    model.reset()
    '''
    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    '''

    all_prediction = {}

    with open('data/test_dials.json', 'r') as f:
        test = json.loads(f.read())

        for d in tqdm(test):


            id = d["dialogue_idx"]
            dia_dict = defaultdict(dict)
            history = []
            tmp = defaultdict(list)
            prev_dict = {}
            model.reset()
            for i,dialogue in enumerate(d['dialogue']):
                if dialogue['system_transcript'] != "":
                    history.append(dialogue['system_transcript'])
                history.append(dialogue["transcript"])
                dom, cs_d = model.predict(history)
                prev_dict = combine_dict(prev_dict, dom, cs_d)
                tmp["pred_bs_ptr"] = convert_cs(prev_dict)
                turn_dict = fix_general_label_error(dialogue["belief_state"], False, SLOTS)
                tmp["turn_belief"] = [str(k)+'-'+str(v) for k, v in turn_dict.items()]
                dia_dict[str(i)] = copy.deepcopy(tmp)




            all_prediction[id] = dia_dict

    with open('data/all_preds_with_book4.json', 'w') as j:

        json.dump(all_prediction, j, indent=4)


if __name__ == "__main__":
    run()
