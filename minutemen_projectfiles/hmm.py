from collections import Counter
with open("twitter_train.txt") as file:
    data = [[tuple(tag.split("\t")) for tag in l.split("\n")] for l in file.read().split("\n\n")]

tagged_words = [tup for sent in data for tup in sent if len(tup) == 2]
word_and_pos_tag_counts = Counter(tagged_words)
word_counts = Counter([tag[0] for tag in tagged_words])
pos_tag_counts = Counter([tag[1] for tag in tagged_words])
words = {word for word,tag in tagged_words}
tags = {tag for word,tag in tagged_words}

def bjw(word, tag):
    delta = 1
    return (word_and_pos_tag_counts[(word, tag)] + delta) / (pos_tag_counts[tag] + delta * (len(words) + 1))

def bjw2(word, tag):
    delta = 1
    return (word_and_pos_tag_counts[(word, tag)] + delta) / (word_counts[word] + delta * (len(words) + 1))

def j_star(word):
    return max((bjw(word, tag), tag) for tag in tags)

def j_star2(word):
    if word not in words:
        return (1, "@")
    return max((bjw2(word, tag), tag) for tag in tags)

f = open("naive_output_probs.txt","w+")
for word, tag in tagged_words:
    f.write(f"{word} {tag} {bjw(word, tag)}\n")
f.close()

# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    with open(in_test_filename) as file2:
        words = [word for word in file2.read().split("\n") if word]

    f = open(out_prediction_filename,"w+")
    for word in words:
        f.write(f"{j_star(word)[1]}\n")
    f.close()

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    with open(in_test_filename) as file:
        words = [word for word in file.read().split("\n") if word]

    f = open(out_prediction_filename,"w+")
    for word in words:
        f.write(f"{j_star2(word)[1]}\n")
    f.close()

# compute Output Probability
def bjw3(word, tag):
    delta = 0.1
    return (word_and_pos_tag_counts[(word, tag)] + delta) / (pos_tag_counts[tag] + delta * (len(words) + 1))

f = open("output_probs.txt","w+")
for word, tag in tagged_words:
    f.write(f"{word} {tag} {bjw3(word, tag)}\n")
f.close()

# compute Transition Probability
def aij(t2, t1):
    if (t1 == "START"):
        count_t1 = len(data) - 1
        count_t2_t1 = 0
        for sent in data[:-1]:
            if sent[0][1] == t2:
                count_t2_t1 += 1

    else:
        count_t1 = pos_tag_counts[t1]
        count_t2_t1 = 0
        for sent in data[:-1]:
            for index in range(len(sent)-1):
                if sent[index][1] == t1 and sent[index+1][1] == t2:
                    count_t2_t1 += 1
    
    delta = 10
    return (count_t2_t1 + delta)/(count_t1 + delta * (len(tags) + 1))

# creating t x t transition matrix of tags, t = no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
import pandas as pd
import numpy as np

list_tags = list(tags)
list_tags.append("START")
tags_matrix = np.zeros((len(list_tags), len(list_tags)), dtype='float32')

for i, t1 in enumerate(list_tags):
    for j, t2 in enumerate(list_tags): 
        tags_matrix[i, j] = aij(t2, t1)

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list_tags, index=list_tags)
tags_df.to_csv('trans_probs.txt', header=list_tags, index=list_tags, sep=' ', mode='a')

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    with open(in_tags_filename) as tagsfile:
        tags = [tag for tag in tagsfile.read().split("\n") if tag]

    with open(in_test_filename) as file2:
        tweets = [sentence.split("\n") for sentence in file2.read().split("\n\n") if sentence]

    f = open(out_predictions_filename,"w+")
    for tweet in tweets:

        bp_matrix = {tag : [(0, "") for word in tweet] for tag in tags}
        
        for (i, word) in enumerate(tweet):
            if i == 0:
                for tag in tags:
                    transition_p = tags_df.loc["START", tag]
                    emission_p = bjw3(word, tag)
                    prob = emission_p * transition_p
                    bp_matrix[tag][i] = (prob, "START")

            else:
                for tag in tags:
                    p = []
                    for prev_tag in tags:
                        transition_p = tags_df.loc[prev_tag, tag]
                        emission_p = bjw3(word, tag)
                        prob = emission_p * transition_p * bp_matrix[prev_tag][i - 1][0]
                        p.append((prob, prev_tag))
                    max_prob = max(p)
                    bp_matrix[tag][i] = max_prob

        final_bp = max((v[-1], k) for k, v in bp_matrix.items())
        final_bp_list = []
        final_bp_list.append(final_bp[1])
        
        i = len(tweet) - 1
        while i > 0:
            next_bp = bp_matrix[final_bp_list[-1]][i][1]
            final_bp_list.append(next_bp)
            i -= 1

        for final_tag in final_bp_list[::-1]:
            f.write(f"{final_tag}\n")
        
    f.close()

import re
def bjw4(word, tag):

    patterns = {
        r'@\w+' : '@',                 # usernames
        r'#\w+' : '#',                 # hashtags
        r'RT' : '~',                   # retweet
        r'\b((https?|ftp)://)?(www\.)?[A-Za-z0-9]+\.[A-Za-z]{2,}\b' : 'U', # url
        r'^-?[0-9]+(.[0-9]+)?$' : '$', # numbers 
        r'.*(?:es|ed|ing)$' : 'V',     # verb
        r'.*ly$' : 'R',                # adverb
        r'.*(ous|ful|able|ant|ary|ic|ive|less|like|ish|est)$' : 'A', # adjectives
        r'.*\'s$' : 'S',               # possessive nouns
        r'.*s$' : 'N',                 # plural nouns
        r'[A-Z][a-z]+' : '^',          # proper nouns
    }
    
    weight = 1
    if word_counts[word] == 0:
        for pattern, t in patterns.items():
            if re.match(pattern, word) and tag == t:
                weight = 100
            if tag == 'N':
                weight = 15

    delta = 0.01
    return weight * (word_and_pos_tag_counts[(word, tag)] + delta) / (pos_tag_counts[tag] + delta * (len(words) + 1))

f = open("output_probs2.txt","w+")
for word, tag in tagged_words:
    f.write(f"{word} {tag} {bjw4(word, tag)}\n")
f.close()
tags_df.to_csv('trans_probs2.txt', header=list_tags, index=list_tags, sep=' ', mode='a')

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    with open(in_tags_filename) as tagsfile:
        tags = [tag for tag in tagsfile.read().split("\n") if tag]

    with open(in_test_filename) as file2:
        tweets = [sentence.split("\n") for sentence in file2.read().split("\n\n") if sentence]
    
    f = open(out_predictions_filename,"w+")
    for tweet in tweets:

        bp_matrix = {tag : [(0, "") for word in tweet] for tag in tags}
        
        for (i, word) in enumerate(tweet):
            if i == 0:
                for tag in tags:
                    transition_p = tags_df.loc["START", tag]
                    emission_p = bjw4(word, tag)
                    prob = emission_p * transition_p
                    bp_matrix[tag][i] = (prob, "START")

            else:
                for tag in tags:
                    p = []
                    for prev_tag in tags:
                        transition_p = tags_df.loc[prev_tag, tag]
                        emission_p = bjw4(word, tag)
                        prob = emission_p * transition_p * bp_matrix[prev_tag][i - 1][0]
                        p.append((prob, prev_tag))
                    max_prob = max(p)
                    bp_matrix[tag][i] = max_prob

        final_bp = max((v[-1], k) for k, v in bp_matrix.items())
        final_bp_list = []
        final_bp_list.append(final_bp[1])
        
        i = len(tweet) - 1
        while i > 0:
            next_bp = bp_matrix[final_bp_list[-1]][i][1]
            final_bp_list.append(next_bp)
            i -= 1

        for final_bp_tag in final_bp_list[::-1]:
            f.write(f"{final_bp_tag}\n")
        
    f.close()


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    wrong = []
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
        else: 
            wrong.append((pred, truth))

    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '/Users/caleb/Desktop/BT3102/Project/projectfiles' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    
    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')


if __name__ == '__main__':
    run()
