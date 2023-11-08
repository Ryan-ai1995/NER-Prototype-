import time
from collections import defaultdict
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk import word_tokenize
import nltk
import re
import requests
import json


def fetch_book_from_url(url):

    # Try to fetch book from URL address
    try:
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:

            # The content of the .txt file is in response.text
            text_content = response.text

        else:
            print(f"Failed to fetch URL. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return text_content


def text_chunk_cleaner(tokenized_chunk_words_only):

    # Filter non-alphanumeric characters from string
    def is_alphanumeric(char):
        return char.isalnum()

    # Apply the filter function to each string in the list
    cleaned_list = [''.join(filter(is_alphanumeric, item))
                    for item in tokenized_chunk_words_only if any(filter(is_alphanumeric, item))]

    # Check if string is a Roman Numeral
    def is_roman_numeral(s):
        
        # Define a pattern to match Roman numerals
        roman_numeral_pattern = r'^(M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|[0-9]+)$'
        return bool(re.match(roman_numeral_pattern, s))

    filtered_list = [
        item for item in cleaned_list if not is_roman_numeral(item)]

    # Filter out strings of length 1 that are not 'a' or 'i'
    filtered_list = [item.lower() for item in filtered_list if len(
        item) != 1 or item in ['a', 'i'] or item.isdigit()]

    # Filter out strings starting with 'www'
    tokenized_chunk_words_only = [
        item for item in filtered_list if not item.startswith('www')]

    return tokenized_chunk_words_only


def extract_desired_entities(ner_results, list_of_desired_entities):

    desired_entities = []

    for index, ner_result in enumerate(ner_results):

        # Check if the entity is of an unwanted type. If it is, remove it from
        # the entity list for this text chunk
        if ner_result['entity_group'] not in list_of_desired_entities:
            continue

        else:
            # Remove any entities that begin with just one letter
            if len(ner_result['word']) == 1:
                continue

            # Check if there is a next entity after the current one that
            # may need to be combined due to '##' in its word.
            if 0 <= index + 1 < len(ner_results):

                for i in range(index+1, len(ner_results)):

                    if ner_results[i]['word'].startswith("##"):
                        ner_result['word'] += ner_results[i]['word'][2:]

                    else:
                        break

                desired_entities.append(ner_result)

            else:
                desired_entities.append(ner_result)

    return desired_entities


def entity_separator(desired_entities):

    # Define two new lists, one for entities of persons and the other
    # for entities of locations
    person_entities = []
    location_entities = []

    for desired_entity in desired_entities:

        if desired_entity['entity_group'] == 'PER':
            person_entities.append(desired_entity)

        else:
            location_entities.append(desired_entity)

    return person_entities, location_entities


def get_chunk_location_indices(location_entities, tokenized_chunk_words_only, num_tokens_processed):

    location_indices = {}

    for location_entity in location_entities:

        if location_entity['word'].lower() in location_indices:
            continue

        # Define entity word and put it in lowercase as text chunk words
        # are also in lowercase
        location_entity_word_lower = location_entity['word'].lower()
        location_entity_word_lower_tokens = word_tokenize(
            location_entity_word_lower)

        for i in range(len(tokenized_chunk_words_only) - len(location_entity_word_lower_tokens) + 1):
            if tokenized_chunk_words_only[i:i+len(location_entity_word_lower_tokens)] == location_entity_word_lower_tokens:

                if location_entity_word_lower not in location_indices:
                    location_indices[location_entity_word_lower] = [
                        i + num_tokens_processed]

                else:
                    location_indices[location_entity_word_lower].append(
                        i + num_tokens_processed)

    return location_indices


def get_chunk_person_index(person_entity, person_index_list_overall, tokenized_chunk_words_only):

    person_index = {}
    person_index_list = []

    # Define entity word and put it in lowercase as text chunk words
    # are also in lowercase
    person_entity_word_lower = person_entity['word'].lower()
    person_entity_word_lower_tokens = word_tokenize(person_entity_word_lower)

    for i in range(len(tokenized_chunk_words_only) - len(person_entity_word_lower_tokens) + 1):
        if i in person_index_list_overall:
            continue

        if tokenized_chunk_words_only[i:i+len(person_entity_word_lower_tokens)] == person_entity_word_lower_tokens:

            if person_entity_word_lower not in person_index:
                person_index[person_entity_word_lower] = i
                person_index_list.append(i)
                break

            else:
                person_index[person_entity_word_lower] = i
                break

    return person_index, person_index_list


def location_indices_dict_combiner(location_indices, location_indices_next):
    
    # Create new dictionary 
    combined_location_indices_dict = defaultdict(list)
    
    # Iterate through the first dictionary and place keys and values in the
    # new dictionary
    for key, value in location_indices.items():
        combined_location_indices_dict[key] += value
        
    # Iterate through the second dictionary and place keys and values in the
    # new dictionary
    for key, value in location_indices_next.items():
        combined_location_indices_dict[key] += value
        
    # Convert back to dictionary variable type
    combined_location_indices_dict = dict(combined_location_indices_dict)

    return combined_location_indices_dict


def get_location_entities_in_search_range(location_entity_name, location_entity_name_indices, person_entity_search_starting_index, person_entity_search_ending_index):

    # Check if any integers in the list are within the specified range
    numbers_within_range = [
        num for num in location_entity_name_indices if person_entity_search_starting_index <= num <= person_entity_search_ending_index]

    if numbers_within_range:
        return [{'name': location_entity_name, 'count': len(numbers_within_range)}]

    else:
        return []
    

def ner_chunk_processor(chunk, num_tokens_processed, nlp, list_of_desired_entities):

    # Tokenize text chunk using NLTK tokenizer to obtain the words
    # within the chunk. Note that we do not mean tokens. The idea here
    # is to essentially extract only the actual words in this text chunk
    # for use later when we need to do person - location matching.
    tokenized_chunk_words_only = word_tokenize(chunk)

    # Perform cleaning and removal of tokens not defined to be words.
    # This is subjective and would require further discussion.
    tokenized_chunk_words_only = text_chunk_cleaner(tokenized_chunk_words_only)

    # Check number of words within this chunk of the text
    num_chunk_tokens = len(tokenized_chunk_words_only)

    # Now, we input the text chunk into the model to obtain person names
    # and locations. We use the grouped_entities option here.
    ner_results = nlp(chunk, grouped_entities=True)

    # Next, we combine entities that are shown in the results as split in two.
    # We can also remove any unwanted entities that we do not need to consider,
    # such as miscellaneous and organisations.
    desired_entities = extract_desired_entities(ner_results, list_of_desired_entities)

    # Now that we have a list of desired entities and also a list of words
    # within this specific text chunk, we will perform a matching process
    # in order to obtain the index values of each entity within the text chunk.
    # Lets separate the entities into two lists now, one for people names and
    # one for locations.
    person_entities, location_entities = entity_separator(desired_entities)

    # Now, we iterate through the location entities list and we obtain the
    # indices for each entity which are the word locations within this
    # text chunk. We add these indices into a list and importantly,
    # we cannot use the same indices twice.
    location_indices = get_chunk_location_indices(
        location_entities, tokenized_chunk_words_only, num_tokens_processed)

    return person_entities, location_entities, location_indices, num_chunk_tokens, tokenized_chunk_words_only


def book_parser(request):
    
    # Extract request fields
    url = request['URL']
    author = request['author']
    title = request['title']
    
    # Import Python Packages
    start_time = time.time()
        
    # Fetch book from provided url
    text_content = fetch_book_from_url(url)
    
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    
    # Instantiate the model pipeline with specified settings
    nlp = pipeline("ner", model=model, tokenizer=tokenizer,
                   aggregation_strategy="simple")
    
    # Run model for all text partitions (there is a token size limit, so partitioning is necessary)
    start_index = 0
    chunk_size = 2000
    
    # Define list of entities that we wish to consider
    list_of_desired_entities = ["PER", "LOC"]
    
    previous_tokenized_chunk_words_only = []
    location_indices_overall = {}
    person_index_list_overall = []
    
    # Define output variables
    person_entity_frequency_dict = {}
    person_to_location_entity_association_dict = {}
    
    # Set the number of tokens processed in total to 0 to start with
    num_tokens_processed = 0
    
    while start_index < len(text_content):
    
        # Define last character index in chunk
        end_index = start_index + chunk_size
    
        # Create text chunk for input into the model
        chunk = text_content[start_index:end_index]
    
        person_entities, location_entities, location_indices, num_chunk_tokens, tokenized_chunk_words_only = ner_chunk_processor(
            chunk, num_tokens_processed, nlp, list_of_desired_entities)
    
        # Combine any previous location indices with the new ones
        location_indices = location_indices_dict_combiner(
            location_indices, location_indices_overall)
    
        # Pick up any previous tokenized words from previous chunks
        previous_tokenized_chunk_words_only.extend(tokenized_chunk_words_only)
        
        # Define the tokenized word list from the text to be all of the previous
        # words as well as the words from the current chunk of text. This helps us
        # keep track of word locations in case we need them during person - location
        # entity association.
        tokenized_chunk_words_only = previous_tokenized_chunk_words_only
        
        # Keep track of the number of words/tokens processed so far. Note that this 
        # number is counted by using the NLTK tokenizer on the text and taking
        # the length of the resulting list of tokens.
        num_tokens_processed = num_tokens_processed + num_chunk_tokens
    
        # Next, we begin to iterate through the person entities, one by one. We
        # obtain the entities indices within the text chunk word list and then
        # we define a search range based upon those indices of UP TO 100 words
        # behind and 100 words ahead.
        for person_entity in person_entities:
            person_index, person_index_list = get_chunk_person_index(
                person_entity, person_index_list_overall, tokenized_chunk_words_only)
            
            # In some cases a person entity provided by the deep learning model cannot
            # be exactly matched and found within the word list. This is likely 
            # because the word list is created using the NLTK tokenizer, while the 
            # deep learning model uses a different tokenizer, and so the resulting
            # token sets are slightly different. We currently continue to the next
            # person entity if we cannot find the exact match and hence we cannot 
            # know its index location within the text.
            if not person_index:
                continue
            
            # Here, we keep track of all of the index locations within the text 
            # that have already been accounted for. I.e. If the person entity name
            # is 'Mina' and we have already recorded the index location where that 
            # name is found in the text, if we get another person entity that is also
            # called 'Mina' we do not want to use the same index location twice. 
            # Instead, we look for the next occurrence of the word 'Mina'
            person_index_list_overall.extend(person_index_list)
    
            # We will add this person entity into our frequency dictionary to keep
            # a record of the number of each unique person entities found in the
            # overall text
            if person_entity['word'].lower() not in person_entity_frequency_dict:
                person_entity_frequency_dict[person_entity['word'].lower()] = 1
    
            else:
                person_entity_frequency_dict[person_entity['word'].lower(
                )] = person_entity_frequency_dict[person_entity['word'].lower()] + 1
    
            # Now that we have the person entity word index in the text chunk,
            # we can define a search range as stated above. Note that for the first
            # (and last) chunks, the actual search range for some entities cannot be
            # 100 words behind and 100 words ahead as there are not enough words
            # surrounding these beginning and ending entities.
            person_entity_search_starting_index = (
                person_index[person_entity['word'].lower()]) - 100
    
            if isinstance(person_entity_search_starting_index, int) and person_entity_search_starting_index < 0:
                person_entity_search_starting_index = 0
    
            person_entity_search_ending_index = person_index[person_entity['word'].lower(
            )] + 100
            
            # If we calculate that the ending index is at an index that is outside of the current text chunk (longer than its length)
            # then we need to process the next chunk so that we have the location entity indexes to associate with this person entity name.
            if isinstance(person_entity_search_ending_index, int) and person_entity_search_ending_index > num_tokens_processed:

                # Define start index for next chunk
                start_index = end_index + 1
    
                # Define last character index in chunk
                end_index = start_index + chunk_size
    
                # Create text chunk for input into the model
                chunk = text_content[start_index:end_index]
    
                # Process the next text chunk to obtain further location entities and also further person entities
                person_entities_next, location_entities_next, location_indices_next, num_chunk_tokens_next, tokenized_chunk_words_only_next = ner_chunk_processor(
                    chunk, num_tokens_processed, nlp, list_of_desired_entities)
    
                # Count the number of tokens processed so far
                num_tokens_processed = num_tokens_processed + num_chunk_tokens_next
    
                # Add new locations in the next text chunk to the location indices dictionary
                combined_location_indices_dict = location_indices_dict_combiner(
                    location_indices, location_indices_next)
                
                # Now, we perform the person - location entity association process by using our pre-defined search range for the person entity
                # and looking through our location entities that we have found so far in the text. If any of the location entities have index
                # locations that are within the search range of our current person entity, we append these into a dictionary along with their count.
                # Some formatting is still necessary of these results, which will be done later on. 
                for location_entity_name, location_entity_name_indices in combined_location_indices_dict.items():
    
                    association_dict = get_location_entities_in_search_range(location_entity_name, location_entity_name_indices,
                                                                             person_entity_search_starting_index, person_entity_search_ending_index)
                    
                    if not association_dict:
                        continue
                    
                    if person_entity['word'].lower() not in person_to_location_entity_association_dict:
                        person_to_location_entity_association_dict[person_entity['word'].lower()] = {
                            'associated_places': association_dict}
    
                    else:
                        result_flag = False
                        result_list = person_to_location_entity_association_dict[person_entity['word'].lower()]['associated_places'] 
                        
                        for result in result_list:
                            name = result['name']
                            count = result['count']
                            
                            if association_dict[0]['name'] == name:
                                result['count'] = result['count'] + association_dict[0]['count']
                                result_flag = True
                        
                        if result_flag == False:    
                            person_to_location_entity_association_dict[person_entity['word'].lower(
                            )]['associated_places'].extend(association_dict)
    
                # Now, extend the person entities list as we have generated more from the next text chunk
                person_entities.extend(person_entities_next)
    
                # Do the same for the location entities
                location_entities.extend(location_entities_next)
    
                # Do the same for the location indices
                location_indices_overall = combined_location_indices_dict
    
                # Update text chunk tokens list
                tokenized_chunk_words_only = tokenized_chunk_words_only + \
                    tokenized_chunk_words_only_next
    
            else:
                # If for this current person entity, the search range is fully within the current chunk token length, that means
                # that all of the possible location entities that we need to associate with this person entity are currently
                # available to us, along with their chunk word indices, in the location_indices dictionary.
    
                # We now can iterate through our location_indices dictionary for this current text chunk. We check
                # each key of the dictionary, which is a unique location entity within this chunk and we examine its
                # text chunk indices. If any of the location entities have indices that are within the search range
                # for the current person entity, we insert a key, value pair into the association dictionary.
                # The key will be the person entity word and the value will be a list of location entity indices
                # within this chunk.
                for location_entity_name, location_entity_name_indices in location_indices.items():
    
                    association_dict = get_location_entities_in_search_range(location_entity_name, location_entity_name_indices,
                                                                             person_entity_search_starting_index, person_entity_search_ending_index)
                    
                    if not association_dict:
                        continue
                    
                    if person_entity['word'].lower() not in person_to_location_entity_association_dict:
                        person_to_location_entity_association_dict[person_entity['word'].lower()] = {
                            'associated_places': association_dict}
    
                    else:
                        result_flag = False
                        result_list = person_to_location_entity_association_dict[person_entity['word'].lower()]['associated_places'] 
                        
                        for result in result_list:
                            name = result['name']
                            count = result['count']
                            
                            if association_dict[0]['name'] == name:
                                result['count'] = result['count'] + association_dict[0]['count']
                                result_flag = True
                        
                        if result_flag == False:    
                            person_to_location_entity_association_dict[person_entity['word'].lower(
                            )]['associated_places'].extend(association_dict)
        
        # If we have reached this point in the code, we have iterated through all of our person entities that we have found so far
        # in the text, INCLUDING any that we picked up from requiring to process the next text chunk due to a person entities search range
        # being outside of the location indices we had available at that time (the large if statement above). Therefore, we now need to 
        # process the next text chunk. We set the start index to the current end index + 1 and we save the current word token list 
        # that we have found so far for use later.
        start_index = end_index + 1
        previous_tokenized_chunk_words_only = tokenized_chunk_words_only
    
    # Sort all result lists
    for result_key, result_value in person_to_location_entity_association_dict.items():
        result_list = person_to_location_entity_association_dict[result_key]['associated_places']
        sorted_result_list = sorted(result_list, key=lambda x: x['count'], reverse=True)
        person_to_location_entity_association_dict[result_key]['associated_places'] = sorted_result_list
        
    sorted_person_entity_freq_dict = dict(sorted(person_entity_frequency_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        
    # Combine output dictionaries
    output_list_of_dicts = []
    for person, freq in sorted_person_entity_freq_dict.items():
        if person in person_to_location_entity_association_dict:
            associated_places = person_to_location_entity_association_dict[person]["associated_places"]    
        else:
            associated_places = []
        output_list_of_dicts.append({"name": person.title(), "count": freq, "associated_places": associated_places})
        
    # Create final result dictionary
    output = {"url": url, "title": title, "author": author, "people": output_list_of_dicts}
    
    # Print script execuation time
    end_time = time.time()
    print("Elapsed Time is: ", end_time - start_time)
    
    return output

