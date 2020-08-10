#!/usr/bin/env python
# coding: utf-8

#.py version of our class code.

import pandas as pd
import numpy as np
import os
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()


class TwentyQuestions():
    """
    Plays twenty questions.
    """

    def __init__(self, kn_file_name, stats_file_name, sim_measure = 'ours', quick_endgame = False):

        self.kn_file_name = kn_file_name
        self.kn = pd.read_csv(self.kn_file_name)

        self.y = self.kn['animal']
        self.new_row = 0
        self.temp = 0
        self.X = self.kn.loc[:, self.kn.columns != 'animal']
        self.counter = 1
        self.answers = dict()

        # Whether or not to use the fancy endgame_lose() function or to use quick_endgame_lose().
        # Default = False, i.e. use endgame_lose().
        self.quick_endgame = quick_endgame

        # By default we will be using our similarity measure. Options are 'ours' or 'corr'
        self.sim_measure = sim_measure

        # Dataframe with our statistics
        # stats consist of only two columns:
            #n_questions: the first one contains the number of questions asked in that play
            #result: the second one containts the result. 0 for a win, 1 for a loss in which the animal was already in the dataset and 2 for a new animal
        self.stats_file_name = stats_file_name

        # If the file doesn't already exist in current directory, create it.
        if self.stats_file_name not in set(os.listdir()):
            with open(self.stats_file_name, 'w') as f:
                f.write('n_questions,result\n')

        self.stats = pd.read_csv(self.stats_file_name)

        # Initialise the "probability" distribution over y: a uniform prior of 20 (arbitrary number) per animal.
        self.y_probdist = pd.DataFrame(self.y)
        self.y_probdist['prob'] = np.repeat(20, len(self.y))
        self.y_probdist = self.y_probdist.set_index('animal')['prob']

        # Records which features are subjective, rather subjective, and objective (we will ask about objective features first).
        self.subj_feats = ['furry', 'smelly', 'smart', 'a_long_life', 'fast', 'slow', 'useful_to_humans', 'sleep_a_lot',
                           'dangerous']
        self.rather_subj_feats = ['have_hair', 'predator', 'venomous', 'type_of_pet', 'black', 'white-colored', 'blue-colored', 'a_brown_color', 'gray-colored', 'spots',                                   'hooves', 'paws', 'in_a_group', 'endangered', 'on_a_farm', 'in_safari_areas', 'in_the_ocean',                                   'commonly_eaten', 'bigger_than_a_microwave']
        self.obj_feats = ['feathers', 'produce_eggs', 'produce_milk', 'fly', 'swim', 'have_teeth', 'have_a_backbone', 'take_breaths', 'fins', 'zero_legs',                           'two_legs', 'four_legs', 'five_legs', 'six_legs', 'eight_legs', 'have_a_tail', 'type_of_mammal', 'type_of_bird', 'reptile', 'type_of_fish',                           'type_of_amphibian', 'type_of_insect', 'pink-colored', 'black_and_white', 'orange-colored', 'red', 'green', 'yellow', 'stripes',
                          'horns_or_antlers', 'have_tusks', 'active_mainly_at_night', 'have_a_shell', 'sting', 'in_a_cold_climate']

    # ====================================================
    # Utility functions to describe some objects and debug
    # ====================================================

    def describe_knowledge_base(self):
        print('There are {0} objects and {1} features for each object.'.format(self.y.shape[0], self.X.shape[1]))

    def undo_play(self):
        """
        This function will delete the last row of our KB, useful for debugging and keeping only the real play rounds, not the prototyping ones.
        Also deletes last row of the stats df. Don't call reset_stats() before calling undo_play().
        Arg:

        Returns:

        """
        #
        last_index_kb = self.kn.shape[0] -1
        self.kn = self.kn.drop(index = last_index_kb)
        last_index_stats = self.stats.shape[0]-1
        self.stats = self.stats.drop(index = last_index_stats)
        self.save_progress()

    def save_progress(self):
        """
        Saves the current state of kb and stats
        Arg:

        Returns:

        """
        #making sure we don't have duplicated rows
        self.kn = self.kn.drop_duplicates().copy()
        self.kn.to_csv(self.kn_file_name, index = False)
        self.stats.to_csv(self.stats_file_name, index = False)

    def reset_stats(self):
        """
        Function to reset the stats dataframe in case it gets messy
        Arg:

        Returns:

        """
        self.stats = pd.DataFrame({'n_questions':[], 'result':[]})
        self.save_progress()


    # ====================================================
    # The following methods are for sampling the feature to ask about in each stage of the game, based on the
    # split cardinality ratio.
    # ====================================================


    def get_distinguishing_feats(self):
        """
        Ranks the features in X in ascending order of abs(1-SCR) and filters out those that contain either all 0s or all 1s
        (i.e. those that cannot be used to distinguish between objects).

        Arg:
        Returns:
            A pandas series of features ranked by abs(1-SCR) ascending, with non-distinguishing features removed.
        """
        # Rank the features, drop the NaNs that were put there by dist_from_1(), and return what remains.
        ranked = self.rank_features()
        distinguishing_feats = ranked.dropna()
        return distinguishing_feats

    def rank_features(self):
        """
        Ranks all features in df by their increasing absolute distance from 1 of the SCR.

        Arg:
        Returns:
            A pandas series of features ranked by abs(1-SCR) ascending
        """
        return self.X.apply(self.dist_from_1).sort_values()

    def dist_from_1(self, feat_col):
        """
        Returns the absolute distance from 1 of the split cardinality ratio for the given column of X.

        Arg:
            feat_col: a pandas series, one column in the data frame.
        Returns:
            A float if there are both 0s and 1s in the column, else np.nan
        """
        counts = feat_col.value_counts()
        if len(counts) == 2:  # i.e. if there are both 1s and 0s in the column
            ratio = counts[0] / counts[1]
            return abs( 1 - ratio )
        return np.nan  # Features that get NaNs are filtered out later :)

    def sample_feature(self, distinguishing_feats):
        """
        Ranks the features in X, creates a probability distribution from the ranking, and samples a feature
        according to this probability distribution, returning this as the feature to ask about.

        Arg:
            distinguishing_feats: pandas series of features ranked by abs(1-SCR) ascending, with non-distinguishing
                                  features removed.
        Returns:
            sampled_feat: A string, the sampled feature to ask about.
        """

        # Get the max value of the distinguishing features (this is the final element, since they're ranked ascending).
        max_val = distinguishing_feats[-1]

        # Subtract each value in the series from max_val+1; now the features will be sorted descending, and the best features
        # to split on will have the highest values.
        # (the +1 is there because otherwise the final feature will have probability 0, and we still want it to be eligible,
        # if improbable)
        ranked_feats_transf = max_val - distinguishing_feats + 1

        # Convert to a probability distribution by dividing by the sum of all observations.
        feat_prob_dist = ranked_feats_transf / ranked_feats_transf.sum()

        # Sample one feature from this distribution and return that feature.
        sampled_feat = np.random.choice( feat_prob_dist.index, 1, p = feat_prob_dist )
        sampled_feat = str(sampled_feat[0])

        return sampled_feat


    # ====================================================
    # The following methods are for asking the user about the sampled feature, getting their answer, splitting the
    # input space accordingly, and updating the probability distribution over animals.
    # ====================================================

    def get_majority_value_and_extremeness(self, feature):
        """
        Looks at how the values are distributed in the given feature. For use in choosing whether to ask
        an unbiased question or a biased question.

        Args:
            X: pandas dataframe with features as columns, populated by 0s and 1s, one row per instance
            feature: a string, the feature we care about.
        Returns:
            majority: integer, 0 or 1, representing majority value for the given feature
            dist_from_equilibrium: float between 0 and 1, representing how out-of-balanced the values for that feature are
            (a value close to 1 means that one value completely overpowers the other; a value closer to 0 means that they
            are better balanced).
        """
        # Count the number of times 0 and 1 each appear in the column and set the more frequent one as majority.
        counts = self.X[feature].value_counts()
        majority = counts.idxmax()

        # Now compute the percentage of ones and determine how far that percentage is from a 50/50 balance. (Multiplied by
        # 2 so that the output distance is in [0, 1], not [0, 0.5] because I think that's more intuitive).
        # See interpretation in docstring.

        if len(counts) == 2: # i.e. if there are both 1s and 0s in the column
            percent_ones = counts[1] / (counts[0] + counts[1])
            dist_from_equilibrium = 2 * abs( percent_ones - 0.5 )

        else: # if only 0s or only 1s in the column; totally out of balance.
            dist_from_equilibrium = 1

        return majority, dist_from_equilibrium

    def ask_and_get_answer(self, feature, majority_val, extremeness):
        """
        Prints question about the supplied feature and gets the answer (checks validity of input).

        Args:
            feature: a string, a column in df
            majority_val: integer, 0 or 1, representing majority value for the given feature.
            extremeness: float between 0 and 1, representing how out-of-balanced the values for that feature are.
        Returns:
            integer in 0, 1, 2 representing the user's answer
        """
        # Asks and gets answer.
        self.ask_about_feature(feature, majority_val, extremeness)
        answ_raw = input()

        # Checks for bad input.
        while answ_raw not in set(['0', '1', '2']):
            print('Please give valid input (0=no, 1=yes, 2=unknown).')
            answ_raw = input()

        # Convert string input to integer (0, 1, or 2) and return.
        answ = int(answ_raw)
        return answ

    def process_answer(self, feature, answ):
        """
        Splits X based on user's answer, adds the answer to the answers dictionary, and modifies the probability
        distribution over animals based on the answer.

        Args:
            feature: a string, a column in df
            answ: integer in 0, 1, 2 representing the user's answer
        Returns:
        """
        # Add answer to the answers database
        self.answers[feature] = answ

        # If the answer is 0 or 1, split dataset, returning only those instances where the answer holds, and update
        # the probability distribution over animals accordingly.
        if answ == 0:
            self.update_animal_probdist(feature, 0)
            self.split_df_on_feature(feature, answ)
        elif answ == 1:
            self.update_animal_probdist(feature, 1)
            self.split_df_on_feature(feature, answ)

        # If the answer is 2, only remove the feature from the dataset; don't split dataset and don't update probdist.
        else:
            self.X = self.X.drop(columns=[feature])

    def split_df_on_feature(self, feature, answer):
        """
        Returns subset of df where df[feature]==answer and drops feature from columns in df.

        Args:
            feature: string, the column name to split on
            answer: int, 0 or 1, reflecting which subset of the dataframe to keep
        Returns:
            pandas dataframe with features as columns (subset of df).
        """
        self.X = self.X[self.X[feature] == answer].drop(columns=[feature])

    def update_animal_probdist(self, feature_asked, answ):
        """
        Given a user's answers to a question about a particular feature, update the probability distribution over animals.

        Args:
            feature_asked: a string, the feature just asked about
            answ: an integer, the user's response
        Returns:
        """

        # Set the index of kn to the animal column for easy combination with the probability distribution.
        kn = self.kn.set_index('animal')

        # Extract the column in kn corresponding to the feature we asked about.
        kn_col = pd.Series(kn[feature_asked])  # Copy this column before modifying it, so that we don't modify kn!

        # Halve current value if incompatible.
        # To do this, replace all wrong answers with 2s and correct answers by 1s,
        # and divide by kn_col (divides mismatches by 2 and matches by 1, i.e. matches stay same)
        if answ == 1:
            kn_col = np.where(kn_col == 0, 2, kn_col)
            self.y_probdist = self.y_probdist / kn_col
        elif answ == 0:
            kn_col = np.where(kn_col == 1, 2, 1)
            self.y_probdist = self.y_probdist / kn_col

    def ask_about_feature(self, feat_name, majority_val, extremeness):
        """
        This function prints out a natural language question (either biased or not) based on the feature name,
        e.g. biased positive: "Your animal does have wings, doesn't it?", non-biased: "Does your animal have wings?".
        No biased negative because of ambiguity of answer (what does it mean to answer 'no' to 'Your animal is yellow,
        isn't it?').

        Args:
            feat_name: string, name of feature to split dataset on
            majority_val: integer, 0 or 1, representing majority value for the given feature.
            extremeness: float between 0 and 1, representing how out-of-balanced the values for that feature are.
        Prints:
            A string, the natural language question asking about that feature.
        Returns:
            Nothing.
        """

        # Initialize bias threshold
        bias_threshold = 0.65

        # PREPROCESSING OF FEATURE NAMES
        vowels = "aeiou"
        feat_name = feat_name.replace("_"," ") # replace all underscores by blank spaces
        word = feat_name.partition(" ") # splits into a 3-tuple at first space, e.g. ('has', ' ', 'many good friends')

        # convert feature into SpaCy doc (i.e. sequence of tokens)
        doc = nlp(feat_name)
        # print("FEATURE:", doc)

        # The question type is decided based on the POS of the first word in the feature name
        first_token = doc[0]
        token_pos = first_token.pos_
        # print('POS:', token_pos)

        # BIASED POSITIVE QUESTIONS: "Your animal has wings, doesn't it?"
        if majority_val == 1 and extremeness > bias_threshold:
            # print('BIASED QUESTION')

            # Participles, adjectives, adverbs
            if (token_pos == "VERB" and feat_name[-2:] == "ed") or token_pos == "ADJ" or token_pos == "ADV":
                question = f"Your animal is {feat_name}, isn't it?"

            # Plural nouns, nouns preceded by determiner
            elif (token_pos == "NOUN" and feat_name[-1] == "s") or token_pos == "DET" or token_pos == 'NUM':
                question = f"Your animal has {feat_name}, doesn't it?"

            # Singular nouns: distinction of vowel-initial and consonant-initial nouns
            elif token_pos == "NOUN" and feat_name[-1] != "s":
                if feat_name[0].lower() in vowels:
                    question = f"Your animal is an {feat_name}, isn't it?"
                else:
                    question = f"Your animal is a {feat_name}, isn't it?"

            # Verbs: Third person singular, distinction of multiple-element feat_names and single-element feat_names
            # and auxiliaries, i.e. possessive 'has'
            elif token_pos == "VERB":
                if word[2] != "": # if feat_name consists of more than 1 word
#                     question = f"Your animal {lexeme(first_token)[1]} {word[2]}, doesn't it?"
                    ## lexeme() is a terrible generator that always crashes on the first run-through but works fine after :(
                    question = f"Your animal {str(first_token)+'s'} {word[2]}, doesn't it?"
                else: # if feat_name consists of only 1 word
                    question = f"Your animal {str(first_token)+'s'}, doesn't it?"

            # Auxiliaries (i.e. 'have')
            elif token_pos == 'AUX':
                question = f"Your animal has {word[2]}, doesn't it?"

            # Adpositions
            elif token_pos == "ADP":
                question = f"Your animal lives {feat_name}, doesn't it?"

            # In case none of these conditions are triggered (shouldn't happen, but, just in case), give up and
            # ask about the feature name alone
            else:
                question = feat_name+'?'

        # NON-BIASED QUESTION: "Does your animal have wings?"
        else:
            # print('UNBIASED QUESTION')

            if (token_pos == "VERB" and feat_name[-2:]== "ed") or token_pos == "ADJ" or token_pos == "ADV":
                question = f"Is your animal {feat_name}?"

            elif (token_pos == "NOUN" and feat_name[-1] == "s") or token_pos == "DET" or token_pos == 'NUM':
                question = f"Does your animal have {feat_name}?"

            elif token_pos == "NOUN" and feat_name[-1] != "s":
                if feat_name[0].lower() in vowels:
                    question = f"Is your animal an {feat_name}?"
                else:
                    question = f"Is your animal a {feat_name}?"

            elif token_pos == 'AUX':
                question = f'Does your animal have {word[2]}?'

            elif token_pos == "VERB":
                question = f"Does your animal {feat_name}?"

            elif token_pos == "ADP":
                question = f"Does your animal live {feat_name}?"

            else:
                question = feat_name+'?'

        print('Q'+str(self.counter)+': '+question)


    # ====================================================
    # The following methods are for once the features are exhausted. They ask about the animals in order of
    # most likely to least likely.
    # ====================================================

    def guess_objs_from_probdist(self):
        """
        To be used once the dataset cannot be split by features anymore but multiple objects still remain.
        Guesses objects in order of descending probability.

        Args:
        Returns:
            Nothing.
        """
        # Sort values descending, so the highest-probability animals are first.
        self.y_probdist.sort_values(ascending=False, inplace=True)

        # Initialise list to collect names of animals already guessed (this is to avoid asking multiple times about
        # an animal if it is given multiple times in the knowledge base; now, we only guess each animal once,
        # corresponding to its highest-probability instance in y_probdist)
        guessed_animals = set()

        # Go through animal in descending order of probability and guess. Skip the animal if it's already been asked.
        for animal in self.y_probdist.index:
            if self.counter <= 20:
                if animal not in guessed_animals:

                    self.ask_about_object(animal)
                    guessed_animals.add(animal)

                    self.counter += 1

                    # Get user input for answer and check that it's OK.
                    answ_raw = input()
                    # Checks for bad input.
                    while answ_raw not in set(['0', '1', '2']):
                        print('Please give valid input (0=no, 1=yes, 2=unknown).')
                        answ_raw = input()
                    answ = int(answ_raw)

                    if answ == 1:
                        self.endgame_win()
                        break

            # Lose because exceeded 20 questions
            else:
                self.quick_endgame_lose() if self.quick_endgame else self.endgame_lose()
                return

    def ask_about_object(self, obj_name):
        """
        This function prints out a natural language question based on the object name,
        e.g. "Are you thinking of an ocelot?"

        Arg:
            obj_name: string, name of object to guess.
        Prints:
            A string, the natural language question guessing that object.
        Returns:
            Nothing.
        """

        vowels = "aeiou" # Initiate string for vowel/consonant distinction

        # Distinction of vowel-initial and consonant-initial nouns
        if obj_name[0].lower() in vowels:
            question = f"Are you thinking of an {obj_name}?"
        else:
            question = f"Are you thinking of a {obj_name}?"

        print('Q'+str(self.counter)+': '+question)


    # ====================================================
    # The following functions are for the endgame: if the system guesses right, it wins. Otherwise, it loses.
    # ====================================================
    #Auxiliary functions for the endgame_lose phase:

    def sim_ours(x, y):
        """
        This function will count how many values x and y have in common
        Arg:
            The rows we are comparing
        Returns:
            Count of common features
        """
        li = []

        for i, el in enumerate(x):
            if el == y[i]:
                li.append(1)
            else:
                li.append(0)

        return sum(li)

    def sim_argmax(self):
        """
        Using correlation between rows to measure similarity and retrieve index and most similar row

        Arg:
            The new row that we will compare to every pre-existing row
        Returns:
            The index of the argmax and it's similarity value
        """
        self.new_row = pd.Series(data = self.new_row[1:], index = self.kn.set_index('animal').columns)
        corrs = np.asarray(self.kn.set_index('animal').corrwith(self.new_row, axis = 1))
        #getting argmax
        argmax = np.argmax(np.abs(corrs))
        return argmax, corrs[argmax]

    # ====================================================

    def endgame_lose(self):
        """
        Handles the case in which we are not able to guess the user's animal.
        First it asks for the correct answer. If the correct answer was already in the dataset, it will create a new row combining the given answers and the existing data on that animal.
        If the correct answer was not on the dataset, it will create a new row taking into account the answers provided by the user and the most similar pre-existing animal on the database.
        After adding this new information to the kb, it will update the stats file and save the game's progress.
        Arg:

        Returns:
        """
        #======================================

        # EP ADDED
        print(self.answers)

        #Swallowing pride
        print('Dangit, you were too smart for me!')

        #======================================

        #Getting correct answer
        print('Which object were you thinking about?')
        correct_answer = '_'.join(input().lower().split())
        #adding the correct answer to the answers dict
        self.answers[self.y.name] = correct_answer
        #print(self.answers)
        print('Smart choice!')

        #=======================================

        # Getting rid of the answers that received a 2
        self.answers = {k:v for k, v in self.answers.items() if v!=2}

        #=======================================

        # If the correct answer is already in our dataset
        if correct_answer in self.y.unique():
            # If the user's answers contradict our KB we will add a new row to the KB with the new information

            #temporary array to keep the updated row
            correct_answer_index = np.where(self.y==correct_answer)[0][0]
            self.new_row = self.kn.iloc[[correct_answer_index]].copy()

            #update process
            for attribute, value in self.answers.items():
                if type(value) != str: #making sure to not compare the animal name
                    if (value == self.new_row[attribute]).bool() == False: #diff than in our KB
                                self.new_row[attribute] = value
            self.kn = self.kn.append(self.new_row, ignore_index=True)

            #adding result to stats
            new_stats_row = pd.DataFrame({'n_questions':[self.counter-1], 'result':[1]})
            self.stats = self.stats.append(new_stats_row, ignore_index=True)

        #if correct answer is not yet in our dataset
        else:


            #retrieving the row that is already in our KB with the highest similarity to the answers provided by the user.
            #if there is a tie, we will simply grab the values from the first row having this similarity maximum value.

            if self.sim_measure == 'ours':

                #blank new row
                self.new_row = []

                #filling in the new row
                for i, attribute in enumerate(self.kn.columns, 0):
                    if attribute in self.answers.keys(): #knowledge provided by the user
                        self.new_row.append(self.answers[attribute])
                    else:
                        self.new_row.append(993993)

                #using our similarity measure
                #we convert to np array and delete the first value with the string 'animal'
                rows = [np.asarray(self.kn.iloc[i].copy())[1:] for i in range(self.kn.shape[0])]

                #here we store the similarity counts between our new row and every other row in our KB
                sim_counts = [self.sim_ours(rows[i], self.new_row) for i in range(len(rows))]

                #retrieving the row index corresponding to the animal with the highest similarity and retrieving that row
                most_similar_index = np.argmax(sim_counts)
                most_similar_row = self.kn.iloc[most_similar_index].copy()

                #second round filling in the new row with the missing features coming from the most similar existing row
                self.new_row = []

                for i, attribute in enumerate(self.kn.columns, 0):
                    if attribute in self.answers.keys(): #knowledge provided by the user
                        self.new_row.append(self.answers[attribute])
                    else:
                        #for the features that were not provided by the user we will use our similarity measure to interpolate the missing values from the most similar row.
                        self.new_row.append(most_similar_row[i])

            elif self.sim_measure == 'corr':
                #blank new row
                self.new_row = []

                #filling in the new row
                for i, attribute in enumerate(self.kn.columns, 0):
                    if attribute in self.answers.keys(): #knowledge provided by the user
                        self.new_row.append(self.answers[attribute])
                    else:
                        self.new_row.append(993993)
                self.temp = self.new_row
                #using pandas correlation between rows to retrieve similarities
                most_similar_index, value = self.sim_argmax()
                most_similar_row = self.kn.iloc[most_similar_index].copy()
                if value >0:
                    #positive correlation so we'll copy the values from the most similar row
                    #second round filling in the new row with the missing features coming from the most similar existing row
                    self.new_row = []

                    for i, attribute in enumerate(self.kn.columns, 0):
                        if attribute in self.answers.keys(): #knowledge provided by the user
                            self.new_row.append(self.answers[attribute])
                        else:
                            #for the features that were not provided by the user we will use our similarity measure to interpolate the missing values from the most similar row.
                            self.new_row.append(most_similar_row[i])
                elif value <= 0:
                    #negative correlation so we'll invert the values from the most different row
                    self.new_row = []

                    for i, attribute in enumerate(self.kn.columns, 0):
                        if attribute in self.answers.keys(): #knowledge provided by the user
                            self.new_row.append(self.answers[attribute])
                        else:
                            #for the features that were not provided by the user we will use our similarity measure to interpolate the missing values from the most similar row.
                            new_val = 0 if most_similar_row[i]==1 else 1
                            self.new_row.append(new_val)


            #adding it to the KN
            final = dict()
            for i, at in enumerate(self.kn.columns, 0):
                final[at] = self.new_row[i]
            self.kn = self.kn.append(final, ignore_index=True)

            #updating stats
            new_stats_row = pd.DataFrame({'n_questions':[self.counter-1], 'result':[2]})
            self.stats = self.stats.append(new_stats_row, ignore_index=True)


        #Updating our KB file
        self.save_progress()

        # Resetting game-dependent variables so we can play again.
        self.reset_game()


    def endgame_win(self):
        """
            Handles the case in which we guess correctly the user's animal, updates the stats file and saves progress.
        """
        print('Oh yeah! I rock')
        new_stats_row = pd.DataFrame({'n_questions':[self.counter-1], 'result':[0]})
        self.stats = self.stats.append(new_stats_row, ignore_index=True)
        self.save_progress()

        # Resetting game-dependent variables so we can play again.
        self.reset_game()


    def quick_endgame_lose(self):
        """
        Quick version of the losing case for prototyping.
        """
        print('dangit')

        # Resetting game-dependent variables so we can play again.
        self.reset_game()


    def reset_game(self):
        """
        After a game has been won or lost, resets all game-dependent variables to their initial states.
        """
        self.X = self.kn.loc[:, self.kn.columns != 'animal']
        self.counter = 1
        self.answers = dict()
        self.y = self.kn['animal']
        self.y_probdist = pd.DataFrame(self.y)
        self.y_probdist['prob'] = np.repeat(20, len(self.y))
        self.y_probdist = self.y_probdist.set_index('animal')['prob']



    # ====================================================
    # Finally, the following function is a recursive function that plays the game.
    # ====================================================

    def play(self):
        """
        Recursively bisects knowledge base based on user input about whether target object matches the feature.
        Guesses animals in order of their descending probability, given the user's answers.
        """

        # -----------------------------
        # BASE CASE 0: counter > 20
        # -----------------------------
        if self.counter > 20:
            print('TOO MANY QUESTIONS!')
            self.quick_endgame_lose() if self.quick_endgame else self.endgame_lose()
            return

        # -----------------------------
        # BASE CASE 1: Only one row left in the data, so only one object compatible with all the answers thus far.
        # Guess it (at top of probdist) and further objects in order of decreasing probability.
        # -----------------------------

        if len(self.X) == 1:
            print('ONLY ONE OBJECT LEFT!')
            self.guess_objs_from_probdist()  # includes endgame
            return

        # -----------------------------
        # BASE CASE 2: Only one feature left in the data (have asked about all other ones). Will need to ask about that feature,
        # subset the data correspondingly, and then go through all remaining objects in descending order of probability.
        # -----------------------------

        if len(self.X.columns) == 1:
            print('ONLY ONE FEATURE LEFT!')
            feature_to_split_on = self.X.columns[0]
            majority_val, extremeness = self.get_majority_value_and_extremeness(feature_to_split_on)
            answ = self.ask_and_get_answer(feature_to_split_on, majority_val, extremeness)
            self.process_answer(feature_to_split_on, answ)
            self.counter += 1

            # If there are no remaining objects to guess after splitting the data on this feature, then endgame_lose().
            if len(self.X.index) == 0:
                print('NO OBJECTS LEFT TO GUESS!')
                self.quick_endgame_lose() if self.quick_endgame else self.endgame_lose()
                return
            # Otherwise, cycle through all remaining objects until endgame.
            else:
                self.guess_objs_from_probdist()  # includes endgame
                return

        # -----------------------------
        # BASE CASE 3: There are no more distinguishing features at all, so the dataset can't be divided anymore.
        # Will just need to cycle through all remaining objects until endgame.
        # -----------------------------

        disting_feats = self.get_distinguishing_feats()

        # Count the distinguishing features in X (i.e. those that aren't all 0s or all 1s) and cycle through objects
        # if there are none.
        if len( disting_feats ) == 0:
            print('NO MORE DISTINGUISHING FEATURES!')
            self.guess_objs_from_probdist()  # includes endgame
            return

        # -----------------------------
        # RECURSIVE CASE: If we get this far, that means we didn't fall into any of the base cases, so the game can be played!
        # -----------------------------

        # We first want to ask about objective features that distinguish the animals; if there are none, we ask about the
        # "rather subjective" features, and if there are also none, we ask about subjective features.

        # Subset disting_feats for the objective features and sample one from the remaining ranking. If a KeyError
        # or IndexError is raised, there aren't any more objective features in disting_feats.
        try:
            disting_feats_obj = disting_feats.reindex(self.obj_feats).sort_values().dropna()
#             print('SAMPLING AN OBJECTIVE FEATURE')
            feature_to_split_on = self.sample_feature(disting_feats_obj)
        except:

            # Subset disting_feats for the rather_subj features. If a KeyError is raised, there aren't any more rather
            # subjective features in disting_feats either.
            try:
                disting_feats_rathersubj = disting_feats.reindex(self.rather_subj_feats).sort_values().dropna()
#                 print('SAMPLING A "RATHER SUBJECTIVE" FEATURE')
                feature_to_split_on = self.sample_feature(disting_feats_rathersubj)

            # Would finally need to look at the truly subjective features.
            except:
                disting_feats_subj = disting_feats.reindex(self.subj_feats).sort_values().dropna()
#                 print('SAMPLING A SUBJECTIVE FEATURE')
                feature_to_split_on = self.sample_feature(disting_feats_subj)

#         # OLD VERSION: Doesn't worry about objective/subjective features; just chooses one and splits on it.
#         feature_to_split_on = self.sample_feature(disting_feats)

        # print('FEAT:', feature_to_split_on)
        majority_val, extremeness = self.get_majority_value_and_extremeness(feature_to_split_on)
        answ = self.ask_and_get_answer(feature_to_split_on, majority_val, extremeness)
        self.process_answer(feature_to_split_on, answ)
        self.counter += 1

        self.play()
game = TwentyQuestions(kn_file_name = 'knowledge_base.csv', stats_file_name='gameplay_stats.csv', quick_endgame = False)
