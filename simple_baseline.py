"""
Create simple baseline where we just look at the predicted action, and look at the training set existing cases of this
action, and predict the same as in the training set (nearest/exact neighbors).
"""
# TODO make more general for when the number of prior actions is not 4?

import torch
from collections import defaultdict, OrderedDict
import json
from itertools import combinations
import random
from itertools import cycle, islice
import editdistance
import numpy as np
from ego4d.evaluation.lta_metrics import AUED
from tqdm import tqdm


def obtain_full_sequence(next_action, entries_train, dict_clips_train, max_seq=20):
    sequence = []
    while (next_action is not None) and len(sequence) < max_seq:
        info = entries_train[next_action[1][0]]
        sequence.append((info['verb_label'], info['noun_label']))
        actions = dict_clips_train[info['clip_uid']]
        next_action = (info['clip_uid'], actions[next_action[1][1]+1]) if len(actions) > next_action[1][1]+1 else None
    if len(sequence) < max_seq:
        # Repeat until full
        sequence = list(islice(cycle(sequence), max_seq))
    return sequence


def compute_diff_score(x):
    # base_value is used to make the default very large or very small (depending on what we want to find)
    score = np.zeros([len(x), len(x)])
    for i in range(len(x)):
        for j in range(i, len(x)):
            score[i, j] = editdistance.eval(x[i], x[j])
    score = score + score.transpose()
    return score


def return_most_different(x, num_to_select):
    """
    It should be return most different, but seeing that all of them are very different, maybe it is better to return the
    most similar ones, as they are the ones that have higher chances of having some signal.
    """
    # We just return the num_to_select that have the lowest score as a mean (the diagonal affects equally to all)
    diff_score = compute_diff_score(x)
    to_return_idx = diff_score.mean(-1).argsort()[:num_to_select]
    return [x[i] for i in to_return_idx]


def main():
    validate = True

    if validate:
        # Obtained running regression
        path_pred_nouns = '/proj/vondrick/didac/preds_nouns_lta_val.pth'
        path_pred_verbs = '/proj/vondrick/didac/preds_verbs_lta_val.pth'

        path_test_data = '/proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations/' \
                         'fho_lta_val.json'
        path_save = '/proj/vondrick/didac/my_predictions_val.json'

        # Obtaining running forecasting - needs debugging (just create dataset)
        path_labeled_video_paths = '/proj/vondrick/didac/labeled_video_paths_val.pth'
        # Obtained running forecasting
        path_results_baseline = '/proj/vondrick/didac/outputs.json'

    else:
        path_pred_nouns = '/proj/vondrick/didac/preds_nouns_lta.pth'
        path_pred_verbs = '/proj/vondrick/didac/preds_verbs_lta.pth'
        path_test_data = '/proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations/' \
                         'fho_lta_test_unannotated.json'
        path_save = '/proj/vondrick/didac/my_predictions_test.json'
        path_labeled_video_paths = '/proj/vondrick/didac/labeled_video_paths.pth'

        # Obtained running forecasting
        path_results_baseline = '/proj/vondrick/didac/outputs_baseline.json'

    # Collect test data

    labeled_videos = torch.load(path_labeled_video_paths)
    preds_nouns = torch.load(path_pred_nouns)
    preds_verbs = torch.load(path_pred_verbs)

    preds_nouns = preds_nouns.argmax(-1).cpu().numpy()
    preds_verbs = preds_verbs.argmax(-1).cpu().numpy()


    with open(path_test_data, "r") as f:
        entries = json.load(f)['clips']
    id_to_entry_index = {f"{entry['clip_uid']}_{entry['action_idx']}": i for i, entry in enumerate(entries)}



    # TODO undo?
    if validate:
        preds_nouns = np.array([e['noun_label'] for e in entries])
        preds_verbs = np.array([e['verb_label'] for e in entries])




    path_taxonomy = '/proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations/' \
                    'fho_lta_taxonomy.json'
    with open(path_taxonomy, 'r') as f:
        taxonomy = json.load(f)

    input_actions = {}
    for path, clips in labeled_videos:
        predicted_input_nouns = []
        predicted_input_verbs = []
        last_action = 0
        for clip in clips['input_clips']:
            last_action = clip['action_idx']
            id_clip_action = f"{clip['clip_uid']}_{clip['action_idx']}"
            idx = id_to_entry_index[id_clip_action]
            # predicted_input_nouns.append(taxonomy['nouns'][preds_nouns[idx]])
            # predicted_input_verbs.append(taxonomy['verbs'][preds_verbs[idx]])
            predicted_input_nouns.append(preds_nouns[idx])
            predicted_input_verbs.append(preds_verbs[idx])

        sample_id = f"{path.split('/')[-1].split('.')[0]}_{last_action}"
        input_actions[sample_id] = (predicted_input_verbs, predicted_input_nouns)

    clip_id_to_number = defaultdict(int)
    for entry in entries:
        clip_id_to_number[entry['clip_uid']] += 1

    with open(path_results_baseline, "r") as f:
        results = json.load(f)
    actions_not_present = [k for k in results if k not in input_actions]
    assert len(actions_not_present) == 0

    input_actions = {k: v for k, v in input_actions.items() if k in results}


    # Collect training data

    path_train_data = '/proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations/' \
                     'fho_lta_train.json'
    with open(path_train_data, "r") as f:
        entries_train = json.load(f)['clips']
    dict_clips_train = defaultdict(list)
    for i, clip in enumerate(entries_train):
        dict_clips_train[clip['clip_uid']].append((i, clip['action_idx']))
    # It's probably already sorted but just in case
    for clip_id, actions in dict_clips_train.items():
        actions.sort(key=lambda y: y[1])
        dict_clips_train[clip_id] = actions


    min_num_next_actions = 3

    dict_next_clip_action_1 = defaultdict(list)
    dict_next_clip_action_2 = defaultdict(list)
    dict_next_clip_action_3 = defaultdict(list)
    dict_next_clip_action_4 = defaultdict(list)
    for clip_id, actions in dict_clips_train.items():
        previous_action3, previous_action2, previous_action1 = None, None, None
        for i, action in enumerate(actions):
            if i + min_num_next_actions >= len(actions):
                break
            info = entries_train[action[0]]
            key_action = (info['verb_label'], info['noun_label'])
            next_action = (clip_id, actions[i+1])  # if len(actions) > i+1 else None
            dict_next_clip_action_1[key_action].append(next_action)

            if i > 0:
                info2 = entries_train[previous_action1[0]]
                key_action_past2 = (info2['verb_label'], info2['noun_label'])
                key_action_2 = frozenset([key_action, key_action_past2])
                dict_next_clip_action_2[key_action_2].append(next_action)

                if i > 1:
                    info3 = entries_train[previous_action2[0]]
                    key_action_past3 = (info3['verb_label'], info3['noun_label'])
                    key_action_3 = frozenset([key_action, key_action_past2, key_action_past3])
                    dict_next_clip_action_3[key_action_3].append(next_action)

                    if i > 2:
                        info4 = entries_train[previous_action3[0]]
                        key_action_past4 = (info4['verb_label'], info4['noun_label'])
                        key_action_4 = frozenset([key_action, key_action_past2, key_action_past3, key_action_past4])
                        dict_next_clip_action_4[key_action_4].append(next_action)

            previous_action3, previous_action2, previous_action1 = previous_action2, previous_action1, action

    # Predict test data

    # Just to check
    existing = defaultdict(int)
    for k, actions in input_actions.items():
        combination_exists = 0
        for verb, noun in zip(*actions):
            if (verb, noun) in dict_next_clip_action_1:
                combination_exists += 1
        if combination_exists == 0:  # TODO this is completely made up!
            dict_next_clip_action_1[(actions[0][0], actions[1][0])] = [('e11326ca-48c9-433e-b89f-b2c06c96b1b9', (163, 7)), ('b6a88030-b904-4f06-89b4-d2c8eb219bf6', (14, 14)), ('cae37cbc-7ff0-40ea-b3a4-6e6a551f01ab', (75, 1)), ('108e24ea-c3be-4bbc-8aaf-893ff4d018a1', (1393, 58))]
        existing[combination_exists] += 1
    # if existing[0] > 0:
    #     raise RuntimeError('You have to do something about this')

    # At this point, all the sequences are considered equally valid, we will not filter by length. So the length has to
    # be filtered before.

    # List all possibilities, and return the top K=5 that are most diverse between them in terms of loss
    # Make sure the selected ones are good enough, not everything works
    K = 5
    dict_predictions = {}

    for k, actions in tqdm(input_actions.items()):
        possible_sequences = []
        certified_sequences = []  # The more valuable ones

        # First start with 4 actions
        key_action_4 = frozenset([(verb, noun) for verb, noun in zip(*actions)])
        possible_sequences += dict_next_clip_action_4[key_action_4]

        if len(possible_sequences) < K:
            certified_sequences = possible_sequences
            possible_sequences = []

            # Next continue with 3 actions
            for options in list(combinations(range(4), 3)):
                key_action_3 = frozenset([(actions[0][i], actions[1][i]) for i in options])
                possible_sequences += dict_next_clip_action_3[key_action_3]

            if len(possible_sequences) + len(certified_sequences) < K:
                certified_sequences += possible_sequences
                possible_sequences = []

                # Next continue with 2 actions
                for options in list(combinations(range(4), 2)):
                    key_action_2 = frozenset([(actions[0][i], actions[1][i]) for i in options])
                    possible_sequences += dict_next_clip_action_2[key_action_2]

                if len(possible_sequences) + len(certified_sequences) < K:
                    certified_sequences += possible_sequences
                    possible_sequences = []

                    # Next continue with 1 action
                    for verb, noun in zip(*actions):
                        possible_sequences += dict_next_clip_action_1[(verb, noun)]

        # Select among the possible sequences
        num_to_select = K - len(certified_sequences)
        if num_to_select == 1:
            # Randomly chose one
            certified_sequences += [random.choice(possible_sequences)]
            certified_sequences = [obtain_full_sequence(c, entries_train, dict_clips_train) for c in certified_sequences]
        else:
            possible_sequences = [obtain_full_sequence(p, entries_train, dict_clips_train) for p in possible_sequences]
            certified_sequences = [obtain_full_sequence(c, entries_train, dict_clips_train) for c in certified_sequences]
            certified_sequences += return_most_different(possible_sequences, num_to_select)

            if len(certified_sequences) < 5:
                certified_sequences = list(islice(cycle(certified_sequences), 5))

        dict_predictions[k] = {"verb": [[c[0] for c in certified_sequences[i]] for i in range(len(certified_sequences))],
                               "noun": [[c[1] for c in certified_sequences[i]] for i in range(len(certified_sequences))]}

    with open(path_save, 'w') as f:
        json.dump(dict_predictions, f)

    if validate:
        ground_truth = defaultdict(list)
        seen_so_far_per_clip = defaultdict(list)
        for entry in entries:
            entry_id = f"{entry['clip_uid']}_{entry['action_idx']}"
            for previous_clip in seen_so_far_per_clip[entry['clip_uid']]:
                if len(ground_truth[previous_clip]) < 20:
                    ground_truth[previous_clip].append((entry['verb_label'], entry['noun_label']))
            seen_so_far_per_clip[entry['clip_uid']].append(entry_id)

        # We will have more GT than actual predictions because there are no predictions for actions at the beginning
        # (they have to accumulate a context of 4 input actions)
        ground_truth = {k: v for k, v in ground_truth.items() if k in dict_predictions}

        # Sort to make sure both dictionaries have same order
        ground_truth = OrderedDict(sorted(ground_truth.items()))
        dict_predictions = OrderedDict(sorted(dict_predictions.items()))

        labels = np.array(list(ground_truth.values()))
        labels_verb = labels[..., 0][..., None]
        labels_noun = labels[..., 1][..., None]
        labels_action = labels_verb * 1000 + labels_noun

        preds_verb = np.array([v['verb'] for v in dict_predictions.values()])
        preds_noun = np.array([v['noun'] for v in dict_predictions.values()])
        preds_verb = np.swapaxes(preds_verb, 2, 1)  # N, Z, K
        preds_noun = np.swapaxes(preds_noun, 2, 1)  # N, Z, K
        preds_action = preds_verb * 1000 + preds_noun

        result_verb = AUED(preds_verb, labels_verb)
        result_noun = AUED(preds_noun, labels_noun)
        result_action = AUED(preds_action, labels_action)


if __name__ == '__main__':
    main()
