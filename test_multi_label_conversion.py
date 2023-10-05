from collections import defaultdict

from typing import Iterable

from utils import prior_shot_type_to_shot_dsc


def _update_before_and_after_word(mini_labels, before_word, after_word):
    n = len(mini_labels)

    before_word[mini_labels[0]].add(None)
    after_word[mini_labels[-1]].add(None)

    for i in range(n - 1):
        after_word[mini_labels[i]].add(mini_labels[i + 1])
        before_word[mini_labels[i + 1]].add(mini_labels[i])


def get_multi_labels_from_labels(original_labels: Iterable[str]):
    # Create an empty set to store unique words
    unique_mini_labels = set()
    before_word = defaultdict(set)
    after_word = defaultdict(set)

    # Loop through the labels
    for value in original_labels:
        # Split the value into words using "_" as the delimiter
        mini_labels = value.split('_')
        # Update dicts for building exclusive pairs later
        _update_before_and_after_word(mini_labels, before_word, after_word)
        # Add the new mini labels to the unique_mini_labels set
        unique_mini_labels.update(mini_labels)

    for first_word, possible_second_words in after_word.items():
        if len(possible_second_words) == 1:
            second_word = next(iter(possible_second_words))
            if second_word is not None and len(before_word[second_word]) == 1:
                new_mini_label = f"{first_word}_{second_word}"
                unique_mini_labels -= {first_word, second_word}
                unique_mini_labels.add(new_mini_label)

    return unique_mini_labels


def test_multi_label_conversion():
    class_labels = prior_shot_type_to_shot_dsc.values()
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    class_mini_labels = list(get_multi_labels_from_labels(class_labels))
    multi_label2id = {label: i for i, label in enumerate(class_mini_labels)}
    id2multi_label = {i: label for label, i in multi_label2id.items()}

    original_id2multi_id = {id_: [1 if mini_label in label else 0 for mini_label in class_mini_labels] for id_, label in
                            id2label.items()}

    print(f"Unique classes: {list(label2id.keys())}.")
    print(f"Unique mini classes: {list(multi_label2id.keys())}.")

    for original_label, mini_labels in {
        id2label[k]: [id2multi_label[i] for i, x in enumerate(original_id2multi_id[k]) if x] for k, v in
        id2label.items()
    }.items():
        original_split_label = original_label.split("_")
        new_split_label = []
        for mini_label in mini_labels:
            new_split_label.extend(mini_label.split("_"))
        assert set(original_split_label) == set(new_split_label)
