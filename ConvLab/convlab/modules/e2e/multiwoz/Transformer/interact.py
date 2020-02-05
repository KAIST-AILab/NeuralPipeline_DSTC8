import random
import os
import torch
import copy
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
import matplotlib.pyplot as plt
import seaborn
DEFAULT_CUDA_DEVICE=-1
DEFAULT_DIRECTORY = "models"
seaborn.set(font=['AppleMyungjo'], font_scale=2)


def visualize(text, target, attention_density, aux):

    plt.clf()
    fig = plt.figure(figsize=(28,24))
    ax = seaborn.heatmap(attention_density[-len(target):,-(len(target) + len(text)) :-len(target) -2],
        xticklabels=[w for w in text[:-2]],
        yticklabels=[w for w in target])

    ax.invert_yaxis()
    #plt.show()
    plt.title(u'Attention Heatmap')
    file_name = './img/_attention_heatmap_' + aux + ".png"
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close()  # close the figure


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
                           'restaurant': ['food', 'pricerange', 'name', 'area'],
                           'hospital': ['department', 'phone'],
                           'hotel': ['name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type'],
                           'attraction': ['type', 'name', 'area'],
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

    def sample_sequence_v1(self, history, current_output=None):

        special_tokens_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)

        dptok = [special_tokens_id[5]]
        sys = [special_tokens_id[3]]
        eos = [special_tokens_id[1]]

        if current_output is None:
            current_output = []

        cs_dict = {}
        kb_results = {}

        i = 0
        dp_count = 0
        cs_count = 0

        dp = []
        cs = []

        cs_done = 0
        dp_done = 0
        constraints = []
        whole_kb = None
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

            if prev.item() in eos:
                X = ['<bos>'] + ['<usr>'] + [self.tokenizer.decode([x]) for x in history[0]] + ['<cs>'] + [
                    self.tokenizer.decode([x]) for x in cs] + [self.tokenizer.decode([x]) for x in dp]
                Y = [self.tokenizer.decode([x]) for x in current_output]
                attentions_cpu = copy.deepcopy(attentions)
                for j in range(len(attentions_cpu)):
                    for i, attn in enumerate(attentions_cpu[j][0].cpu()):
                        visualize(X, Y, attn, 'v1_res_{}_{}'.format(j, i))
                break
            if prev.item() in dptok:

                if cs_count == 0:

                    X = ['<bos>'] + ['<usr>'] + [self.tokenizer.decode([x]) for x in history[0]]
                    Y = ['<cs>'] + [self.tokenizer.decode([x]) for x in cs]
                    attentions_cpu = copy.deepcopy(attentions)
                    for j in range(len(attentions_cpu)):
                        for i, attn in enumerate(attentions_cpu[j][0].cpu()):
                            visualize(X, Y, attn, 'cs_{}_{}'.format(j, i))
                            pass
                    cs_text = self.decode(cs)

                    if self.cur_dom != cs_text.split(' ')[0] and cs_text.split(' ')[0] in self.domains:
                        self.cur_dom = cs_text.split(' ')[0]

                    keys = self.cs_mapping[self.cur_dom] if self.cur_dom else []

                    if keys != []:

                        prev_key = (0, '')
                        cs_tok = cs_text.split(' ')
                        for j, tok in enumerate(cs_tok):
                            if tok in keys:
                                if prev_key[1] != '':
                                    cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1: j])
                                prev_key = (j, tok)
                            if j == len(cs_tok) - 1:
                                cs_dict[prev_key[1]] = ' '.join(cs_tok[prev_key[0] + 1:])

                    constraints = []
                    cs_key = []

                    for k in cs_dict:
                        if k in self.prev_cs and self.prev_dom == self.cur_dom:
                            if cs_dict[k] in ['<nm>', '', '<nm> '] and cs_dict[k] != self.prev_cs[k]:
                                if cs_dict[k] in ['<dc>', '<dc> ']:
                                    constraints.append([k, 'dontcare'])
                                else:
                                    constraints.append([k, self.prev_cs[k]])
                        else:
                            if not cs_dict[k] in ['<nm>', '', '<nm> ']:
                                if cs_dict[k] in ['<dc>', '<dc> ']:
                                    constraints.append([k, 'dontcare'])
                                else:
                                    constraints.append([k, cs_dict[k]])
                                cs_key.append(k)

                    kb_results = query(self.cur_dom, constraints) if self.cur_dom else None
                    if self.cur_dom == 'train':
                        if 'leaveAt' in cs_key:
                            kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                        else:
                            kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)
                    whole_kb = kb_results
                    kb_results = self.convert_kb(kb_results[0]) if kb_results else None
                    cs_count += 1
                    cs_done += 1
                    i = 0

            if prev.item() in sys:

                if dp_count == 0:

                    X = ['<bos>'] + ['<usr>'] + [self.tokenizer.decode([x]) for x in history[0]] + ['<cs>'] + [
                        self.tokenizer.decode([x]) for x in cs]
                    Y = [self.tokenizer.decode([x]) for x in dp]
                    attentions_cpu = copy.deepcopy(attentions)
                    for j in range(len(attentions_cpu)):
                        for i, attn in enumerate(attentions_cpu[j][0].cpu()):
                            #visualize(X, Y, attn, 'v1_dp_{}_{}'.format(j, i))
                            pass
                    dialog_act = dp[1:]
                    da_text = self.decode(dialog_act)
                    da_dict = self.convert_da(da_text, self.dia_act_dict)
                    da_dict = self.convert_value(da_dict, constraints, None, whole_kb)
                    dp = self.tokenizer.encode(' '.join(self.convert_act(da_dict)))

                    i = 0
                    dp_count += 1
                    dp_done += 1

            if not cs_done:
                cs.append(prev.item())
            elif not dp_done:
                dp.append(prev.item())
            else:
                current_output.append(prev.item())
            i += 1

        self.prev_cs = constraints
        self.prev_dom = self.cur_dom
        return current_output[1:], dp[1:], cs_dict, kb_results, whole_kb

    def sample_sequence_v4(self, history, current_output=None):

        special_tokens_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)

        dptok = [special_tokens_id[5]]
        sys = [special_tokens_id[3]]
        eos = [special_tokens_id[1]]

        if current_output is None:
            current_output = []

        cs_dict = {}
        kb_results = {}

        i = 0
        dp_count = 0
        cs_count = 0

        dp = []
        cs = []

        cs_done = 0
        dp_done = 0
        constraints = []
        whole_kb = None
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

            if prev.item() in eos:

                X = [self.tokenizer.decode([x]) for x in dp]
                Y = [self.tokenizer.decode([x]) for x in current_output]
                attentions_cpu = copy.deepcopy(attentions)
                for j in range(len(attentions_cpu)):
                    for i, attn in enumerate(attentions_cpu[j][0].cpu()):

                        visualize(X, Y, attn, 'res_{}_{}'.format(j,i))
                break

            if prev.item() in dptok:

                X = ['<bos>']+['<usr>'] + [self.tokenizer.decode([x]) for x in history[0]]
                Y = ['<cs>']+[self.tokenizer.decode([x]) for x in cs]
                attentions_cpu = copy.deepcopy(attentions)
                for j in range(len(attentions_cpu)):
                    for i, attn in enumerate(attentions_cpu[j][0].cpu()):
                        pass
                        #visualize(X, Y, attn, 'cs_{}_{}'.format(j,i))
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

                    constraints = []
                    cs_key = []
                    for k in cs_dict:
                        if k in self.prev_cs and self.prev_dom == self.cur_dom:
                            if cs_dict[k] in ['<nm>', '', '<nm> '] and cs_dict[k] != self.prev_cs[k]:
                                if cs_dict[k] in ['<dc>', '<dc> ']:
                                    constraints.append([k, 'dontcare'])
                                else:
                                    constraints.append([k, self.prev_cs[k]])
                        else:
                            if not cs_dict[k] in ['<nm>', '', '<nm> ']:
                                if cs_dict[k] in ['<dc>', '<dc> ']:
                                    constraints.append([k, 'dontcare'])
                                else:
                                    constraints.append([k, cs_dict[k]])
                                cs_key.append(k)
                    kb_results = query(self.cur_dom, constraints) if self.cur_dom else None
                    if self.cur_dom == 'train':
                        if 'leaveAt' in cs_key:
                            kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                        else:
                            kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)
                    whole_kb = kb_results
                    kb_results = self.convert_kb(kb_results[0]) if kb_results else None
                    cs_count += 1
                    cs_done += 1
                    i = 0

            if prev.item() in sys:

                X = ['<ds>'] + [self.tokenizer.decode([x]) for x in cs]
                Y = [self.tokenizer.decode([x]) for x in dp][:4]
                attentions_cpu = copy.deepcopy(attentions)
                for j in range(len(attentions_cpu)):
                    for i, attn in enumerate(attentions_cpu[j][0].cpu()):
                        pass
                        #visualize(X, Y, attn, 'dp_{}_{}'.format(j,i))
                if dp_count == 0:
                    dialog_act = dp[1:]
                    da_text = self.decode(dialog_act).strip()

                    da_tok = da_text.split(' ')
                    toks = []
                    for i, t in enumerate(da_tok):

                        if t in act_name:
                            toks.extend(t[1:-1].split('-'))
                        elif t in slot_name:
                            toks.append(t[1:-1])
                        else:
                            toks.append(t)
                    da_dict = self.convert_da(' '.join(toks), self.dia_act_dict)
                    da_dict = self.convert_value(da_dict, constraints, None, whole_kb)
                    bs = []

                    for d in da_dict:
                        bs.append('<' + d.lower() + '>')
                        for slot, value in da_dict[d]:
                            bs.append('<' + slot.lower() + '>')
                            if isinstance(value, dict):
                                for k in value.keys():
                                    bs.append(k)
                                    bs.append(value[k])
                            else:
                                bs.append(value.lower())
                    dp = self.tokenizer.encode('<dp> '+' '.join(bs))
                    i = 0
                    dp_count += 1
                    dp_done += 1

            if not cs_done:
                cs.append(prev.item())
            elif not dp_done:
                dp.append(prev.item())
            else:
                current_output.append(prev.item())
            i += 1

        self.prev_cs = constraints
        self.prev_dom = self.cur_dom
        return current_output[1:], dp[1:], cs_dict, kb_results, whole_kb



    def convert_da(self, da, dia_act_dict):

        da = da.replace('i d', 'id')
        da_list = da.split(' ')
        for p in range(len(da_list)):
            if p != len(da_list) - 1 and da_list[p] == 'parking' and da_list[p + 1] == 'none':
                da_list[p + 1] = 'yes'

        i = 0
        idlist = []
        while i < len(da_list):
            act = '-'.join(da_list[i:i + 2])
            if act in dia_act_dict.keys():
                idlist.append(i)
            i += 1
        da_dict = {}
        for i in range(len(idlist)):

            act = '-'.join(da_list[idlist[i]:idlist[i] + 2])

            if i == len(idlist) - 1:
                sv = da_list[idlist[i] + 2:]
            else:
                sv = da_list[idlist[i] + 2:idlist[i + 1]]
            sv_id = []
            for slot in dia_act_dict[act]:
                for j in range(len(sv)):
                    if slot == sv[j]:
                        if j > 0 and sv[j - 1] != 'none':
                            sv_id.append(j)
                        if j == 0:
                            sv_id.append(j)

            sv_list = []
            sv_id.sort()
            k = 0
            while k < len(sv_id):

                if k == len(sv_id) - 1:
                    sv_list.append([sv[sv_id[k]], ' '.join(sv[sv_id[k] + 1:])])
                else:
                    sv_list.append([sv[sv_id[k]], ' '.join(sv[sv_id[k] + 1:sv_id[k + 1]])])
                k += 1
            if act in da_dict.keys():
                da_dict[act] += sv_list
            else:

                da_dict[act] = sv_list
        return da_dict

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

    def convert_act(self, dialog_act):

        bs = []
        for d in dialog_act:
            dom, act = d.split('-')
            bs.append(dom.lower())
            bs.append(act.lower())
            for slot, value in dialog_act[d]:
                bs.append(slot.lower())
                if isinstance(value, dict):
                    for k in value.keys():
                        bs.append(k)
                        bs.append(value[k])
                else:
                    bs.append(value.lower())

        return bs

    def convert_value(self, da_dict, constraints, kb, whole_kb):

        if kb is None:
            if 'v1' in self.model_name or 'v4' in self.model_name:
                tmp = {}
                tmp['{}-nooffer'.format(self.cur_dom)] = constraints
                da_dict = tmp
            else:
                da_dict.pop('', None)

        else:

            del_key = []
            for dom_act in da_dict.keys():
                if dom_act == '':
                    del_key.append(dom_act)
                    continue
                if dom_act.split('-')[1] in ['nobook', 'nooffer']:
                    del_key.append(dom_act)
                    continue

                for i, sv in enumerate(da_dict[dom_act]):
                    key = sv[0]
                    if 'hotel' in dom_act and key == 'price':
                        key = 'pricerange'
                    if key in kb.keys():
                        if da_dict[dom_act][i][1] != '?':
                            if not key in ['ref', 'phone', 'id', 'post'] and 'v1' in self.model_name:
                                da_dict[dom_act][i][1] = kb[key]
                            elif not key in ['ref', 'phone', 'id', 'post', 'addr', 'name']:
                                da_dict[dom_act][i][1] = kb[key]

                    elif key == 'area':
                        for area in ["centre", "east", "south", "west", "north"]:
                            if area in sv[1]:
                                da_dict[dom_act][i][1] = area
                    elif key == 'price':
                        for price in ["cheap", "expensive", "moderate", "free"]:
                            if price in sv[1]:
                                da_dict[dom_act][i][1] = price
                    elif key == 'ticket':
                        if 'gbp' in sv[1]:
                            da_dict[dom_act][i][1] = sv[1].replace('gbp', 'pounds')
                    elif key == 'choice':
                        if sv[1].isdigit():
                            da_dict[dom_act][i][1] = str(len(whole_kb))

            for key in del_key:
                if key.split('-')[0] == 'train':
                    da_dict['train-offerbook'] = [['ref', '[train_reference]']]
                elif key.split('-')[0] == 'nooffer':
                    da_dict['{}-inform'.format(self.cur_dom)] = da_dict[key]
                da_dict.pop(key, None)

        return da_dict

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

    def predict(self, usr):

        self.t += 1
        self.history.append(self.tokenizer.encode(usr.lower()))
        with torch.no_grad():
            if 'v1' in self.model_name:
                out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence_v1(self.history)
            else:
                out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence_v4(self.history)

        self.history.append(out_ids)
        out_text = self.decode(out_ids, skip_special_tokens=False)
        out_text = self.postprocess(out_text, kb_results, whole_kb)
        self.history = self.history[-(2 * self.max_history + 1):]

        return out_text

    def postprocess(self, out_text, kb_results, whole_kb):
        # heuristics
        if 'center of town' in out_text:
            out_text = out_text.replace('center of town', 'centre')
        if 'south part of town' in out_text:
            out_text = out_text.replace('south part of town', 'south')
        if 'no entrance fee' in out_text:
            out_text = out_text.replace('no entrance fee', 'free')
        if 'No entrance fee' in out_text:
            out_text = out_text.replace('No entrance fee', 'free')
        sv = ['reference', 'id', 'postcode', 'phone', 'addr', 'name']
        slots = ['[' + self.cur_dom + '_' + s + ']' for s in sv]
        default_value = {'ref': '00000000', 'id': 'tr7075', 'post': 'cb21ab', 'phone': '01223351880', 'name': 'error',
                         'addr': "Hills Rd , Cambridge"}

        for slot, s in zip(slots, sv):

            if s == 'reference':
                t = 'ref'
            elif s == 'postcode':
                t = 'post'
            else:
                t = s

            if slot in slots:

                if out_text.count(slot) > 1:
                    try:
                        if len(kb_results) > 1:

                            out_tok = []
                            tmp = copy.deepcopy(out_text).split(' ')
                            k = 0
                            for tok in tmp:
                                if tok == slot:

                                    out_tok.append(self.convert_kb(whole_kb[k])[t])
                                    k += 1
                                else:
                                    out_tok.append(tok)

                                out_text = ' '.join(out_tok)
                    except:
                        out_text = out_text.replace(slot, default_value[t])

                else:

                    try:
                        if slot == '[taxi_phone]':
                            out_text = out_text.replace(slot, ''.join(kb_results['taxi_phone']))
                        else:
                            out_text = out_text.replace(slot, kb_results[t])
                    except:
                        out_text = out_text.replace(slot, default_value[t])

        return out_text.strip()



def run():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt2-v1', help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint", type=str, default='./models/v1', help="Path, url or short name of the model")
    args = parser.parse_args()

    model = Transformer(model=args.model, model_checkpoint=args.model_checkpoint)
    model.reset()
    '''
    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    '''
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        if raw_text == 'r':
            model.reset()
            continue
        out_text = model.predict(raw_text)
        print('sys:', out_text)

if __name__ == "__main__":
    run()
