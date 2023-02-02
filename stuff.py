import spacy
import cohere
import re

nlp = spacy.load("en_core_web_lg")
co = cohere.Client('FDVePphzMM3EQvpLcA0Fq8fdUhmdwt3OajSnL4js')

STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
nlp.Defaults.stop_words.add("|")

def process_text(text):
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(str(token.lemma_))
    return " ".join(result)

def calculate_similarity(text1, text2):
    base = nlp(process_text(text1))
    compare = nlp(process_text(text2))
    return base.similarity(compare)

def generate_categories(dictionary_with_keys_as_words): 
    categories = []
    for key in dictionary_with_keys_as_words:
        response = co.generate( 
            model='large', 
            prompt=key,
            max_tokens=4, 
            temperature=0.06, 
            k=0, 
            p=1, 
            frequency_penalty=0.61, 
            presence_penalty=0, 
            stop_sequences=["--"], 
            return_likelihoods='NONE') 
        categories.append(format(response.generations[0].text))
    new = []
    for x in categories: 
        temp = re.findall("\w", x)
        temp = "".join(temp)
        new.append(temp)
    return new

def group(dictionary):
    new_dict = {}
    first_value = list(dictionary.values())[0]
    for key in dictionary: 
        if dictionary[key] == first_value: 
            filtered = process_text(dictionary[key])
            new_dict.update({filtered: [key]})
        else: 
            for group in new_dict:
                filtered = process_text(dictionary[key])
                similarity = calculate_similarity(group, filtered)
                if similarity >= 0.53: 
                    new_dict[group].append(key)
                    break
                else: 
                    new_dict[filtered] = [key]
                    break
    return new_dict

def combine(categories, dictionary_with_keys_as_words): 
    dict = {}
    values = list(dictionary_with_keys_as_words.values())
    counter = 0
    for x in categories: 
        dict[x] = values[counter]
        counter += 1
    return dict

def call_all(dict): 
    temp = group(dict)
    categories = generate_categories(temp)
    final = combine(categories, temp)
    return final 

dict = {"id1": "UC Berkeley - Calendar - Week of October 9, 2022", "id2": "Google Calendar - November 2022", "id3": "Explore Microsoft: Intern Opportunities for University Students"}

print(call_all(dict))

