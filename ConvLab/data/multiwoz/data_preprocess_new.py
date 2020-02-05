import json
import copy
import os
from tqdm import tqdm
DOMAIN = ['Attraction', 'Hospital', 'Hotel', 'Police', 'Restaurant', 'Taxi', 'Train']
DOMAIN.extend([name.lower() for name in DOMAIN])

def preprocess(trainval):
    train_path = '{}.json'.format(trainval)
    with open(train_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    no = 0
    iter = 0
    for data_name, data_point in dataset.items():
    # data_point.keys() = ['goal', 'log']
        iter += 1
        if iter % 100 == 0:
            print("preprocess iteration [{}/{}]".format(iter, len(dataset.keys())))

        log = data_point['log']
        goal = data_point['goal']
        count = 0
        cur_domain = None
        prev_domain = None
        for _, turn in enumerate(log):
            text = turn['text']
            dact = turn['dialog_act']
            span = turn['span_info']
            meta = turn['metadata']

            for dom_int, slot_val_list in dact.items():
                dom, intent = dom_int.split('-')
                if intent in ['NoBook']:
                    count = 1
                if dom in DOMAIN and dom != cur_domain:
                    prev_domain = cur_domain
                    cur_domain = dom.lower()
                    

                for sv_idx, slot_val in enumerate(slot_val_list):
                    slot, val = slot_val
                    if slot == 'Addr':
                        if not cur_domain in ['hotel', 'restaurant', 'attraction', 'hospital', 'police']:
                            addr_value = '[{}_addr]'.format(prev_domain)
                        else:
                            addr_value = '[{}_addr]'.format(cur_domain)
                        slot_val_list[sv_idx][1] = addr_value
                        turn['text'] = text.replace(val, addr_value); text = turn['text']
                    if slot == 'Phone':
                        if not cur_domain in ['hotel', 'restaurant', 'attraction', 'taxi', 'hospital', 'police']:
                            phone_value = '[{}_phone]'.format(prev_domain)
                        else:
                            phone_value = '[{}_phone]'.format(cur_domain)
                        slot_val_list[sv_idx][1] = phone_value
                        turn['text'] = text.replace(val, phone_value); text = turn['text']
                    if slot == 'Ref':
                        if not cur_domain in ['hotel', 'restaurant', 'train']:
                            ref_value = '[{}_reference]'.format(prev_domain)
                            print(cur_domain)
                        else:
                            ref_value = '[{}_reference]'.format(cur_domain)
                        slot_val_list[sv_idx][1] = ref_value
                        turn['text'] = text.replace(val, ref_value); text = turn['text']
                    if slot == 'Post':
                        if not cur_domain in ['hotel', 'restaurant', 'attraction', 'hospital']:
                            post_value = '[{}_postcode]'.format(prev_domain)
                        else:
                            post_value = '[{}_postcode]'.format(cur_domain)
                        slot_val_list[sv_idx][1] = post_value
                        turn['text'] = text.replace(val, post_value);
                        text = turn['text']
                    if slot == 'Id':
                        id_value = '[train_id]'
                        slot_val_list[sv_idx][1] = id_value
                        turn['text'] = text.replace(val, id_value);
                        text = turn['text']
                    if slot == 'Name':
                        if not cur_domain in ['hotel', 'restaurant', 'attraction', 'hospital', 'police']:
                            name_value = '[{}_name]'.format(prev_domain)
                        else:
                            name_value = '[{}_name]'.format(cur_domain)
                        slot_val_list[sv_idx][1] = name_value
                        turn['text'] = text.replace(val, name_value);
                        text = turn['text']
        if count == 1:
            no += 1
    outfile_name = '{}_v4.json'.format(trainval)
    print('Writing start')
    print(outfile_name)
    with open(outfile_name, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
    print('Done!')
    print('nobook, nooffer : {}'.format(no))
if __name__ == "__main__":
    preprocess("val")
    preprocess("train")
