#!/usr/bin/env/ conda:"base"
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn

# English is a very ambiguous language. Almost every other word has
# a different meaning in different contexts.


CHAIR: str = "Chair"
ANIMAL: str = "Elephant"
HUMAN: str = "Human"
FRUIT: str = "apple"


def know_senses(word: str) -> None:
    """ Returns definition, synonymous and example of the 
        given word.

        Parameters
        ----------------
        word: string

        Return
        ----------------
        None
    """
    synset = wn.synsets(word)
    for sense in synset:
        print()
        print(f"{sense} : ",
              f"\nDefinition: {sense.definition()}",
              f"\nLemmas/Synonymous: {sense.lemma_names()}",
              f"\nExample: {sense.examples()}"
              )
    return


def hypernyms_hyponemes(word: str) -> None:
    """ hypernyms() API function on the given word say women Synset; 
    it will return the set of synsets that are direct parents of 
    the same.\n
    hypernym_paths() - returns list of sets. Each set contains the 
    path from the root node to the woman Synset

    Parameters:
    ----------
    word: string

    Return:
    ----------
    None
    """
    sense = wn.synset(word)
    hypernym = sense.hypernyms()
    paths = sense.hypernym_paths()
    print(hypernym, paths, sep="\n\n")


if __name__ == "__main__":
    chair_synsets = wn.synsets(CHAIR)
    # print(f"Synsents / senses of chair: \n{chair_synsets}")

    things: list = [CHAIR, ANIMAL, HUMAN, FRUIT]
    print([know_senses(thing) for thing in things])

    word = "woman.n.02"
    hypernyms_hyponemes(word)
