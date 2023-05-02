from quantumGA import QuantumGA
from preproc import Preprocessor
from rouge import Rouge
import os

file_list = [
    '001.txt',
    '002.txt',
    '003.txt',
    '004.txt',
    '007.txt',
    '008.txt',
    '012.txt',
]

rouge_1_r_arr = []
rouge_1_p_arr = []
rouge_1_f_arr = []

rouge_2_r_arr = []
rouge_2_p_arr = []
rouge_2_f_arr = []

rouge_l_r_arr = []
rouge_l_p_arr = []
rouge_l_f_arr = []

category = 'business'

data = {
        'article': '',
        'summaries': '',
        'titles': '',
    }

data_list = []

# article_path = './new_data/article/'+category+'/'
# summaries_path = './new_data/summaries/'+category+'/'
# titles_path = './new_data/titles/'+category+'/'

# for filename in os.listdir(article_path):
#     # Read the contents of the file
#     with open(os.path.join(article_path, filename), 'r', encoding='utf-8') as f:
#         content = f.read().strip()
#         data['article'] = content

# for filename in os.listdir(summaries_path):
#     # Read the contents of the file
#     with open(os.path.join(summaries_path, filename), 'r', encoding='utf-8') as f:
#         content = f.read().strip()
#         data['summaries'] = content

# for filename in os.listdir(titles_path):
#     print("filename = ", filename)
#     # Read the contents of the file
#     with open(os.path.join(titles_path, filename), 'r', encoding='utf-8') as f:
#         content = f.read().strip()
#         data['titles'] = content

#     data_list.append(data.copy())

article_path = './data/bbc_news_data/article/' + category + '/'
summaries_path = './data/bbc_news_data/summaries/' + category + '/'

for filename in os.listdir(article_path):
    # Read the contents of the file
    with open(os.path.join(article_path, filename), 'r', encoding='utf-8') as f:
        data['file_name'] = filename
        data['titles'] = f.readline()
        f.readline()
        content = f.read().strip()
        data['article'] = content
    
    # Read the contents of the file
    with open(os.path.join(summaries_path, filename), 'r', encoding='utf-8') as f:
        content = f.read().strip()
        data['summaries'] = content

    data_list.append(data.copy())


for data in data_list[:50]:
    text = data['article']
    ref_summary = data['summaries']
    doc_title = data['titles']

    print("working on file: ", data['file_name'])
    # print( "text = ", text)
    # print( "ref_summary = ", ref_summary)
    # print( "doc_title = ", doc_title)

    preProc = Preprocessor()
    preProc.preprocessing_text(text)
    preproc_doc_title = preProc.preprocessing_titles(doc_title)

    destination_dir = './'
    directory_name = ['popsize_4']

    quantumGA = QuantumGA(
        preProc.preprocSentences,
        preProc.word_of_sent,
        preProc.preprocTokens,
        preProc.distWordFreq,
        0.5, 0.5,
        userSummLen=250
    )

    quantumGA.tfisf_cosineSim_Calculate()
    quantumGA.cosineSimWithTitle(preproc_doc_title)
    # designed_sizes = [int(preProc.sentencesNum/2)]'
    designed_sizes = [int((((preProc.sentencesNum - 6) / 42) * 50) + 50)]

    for i in range(len(designed_sizes)):
        pop_size = designed_sizes[i]
        # print("pop_size = ", pop_size)

        # With this loop 10 summaries are generated for each topic
        # for index in range(10):
        quantum_pop = quantumGA.initialPop(pop_size)
        # print("quantum_pop = ", quantum_pop)
        quantumGA.measurement(quantum_pop)
        for q_indiv in quantum_pop:
            # print()
            # print("q_indiv = ", q_indiv.binary)
            quantumGA.evalFitness3(q_indiv)

        best_indiv = quantumGA.bestIndividual(quantum_pop)
        # print("best_indiv = ", best_indiv.binary)
        generation = 1
        fitness_change = 0  # The number of consecutive generation that fitness has not changed
        while fitness_change < 20 and generation < 500:
            mating_pool = quantumGA.boltzmann_selection(
                quantum_pop, pop_size,25)
            q_offsprings = quantumGA.twoPointCrossover(mating_pool)
            quantumGA.flipMutation(q_offsprings)
            for offspring in q_offsprings:
                if not offspring.fitness.valid:
                    quantumGA.indivMeasure(offspring)
                    quantumGA.evalFitness3(offspring)
            new_qOffsprings = quantumGA.rotationGate(
                q_offsprings, best_indiv)
            quantumGA.measurement(new_qOffsprings)
            for q_indiv in new_qOffsprings:
                quantumGA.evalFitness3(q_indiv)
            new_qPop = quantumGA.bestReplacement(
                quantum_pop, new_qOffsprings)
            best_indiv = quantumGA.bestIndividual(new_qPop)
            if quantumGA.terminCriterion1(quantum_pop, new_qPop):
                fitness_change += 1
            else:
                fitness_change = 0
            quantum_pop = new_qPop[:]
            generation += 1
        finalSummLen = 0
        summary = ''
        for i in range(len(best_indiv.binary)):
            if best_indiv.binary[i] == 1:
                finalSummLen += len(quantumGA.tokens[i])
                summary += '{}\n'.format(preProc.splitedSent[i])
        summary = summary.rstrip()
        ref_summary = ref_summary.rstrip()

        print("summary = ", summary)

        # Checking summary scores
        rouge = Rouge()
        scores = rouge.get_scores(summary, ref_summary)
        print("scores = ", scores)
        print()

        rouge_1_r_arr.append(scores[0]['rouge-1']['r'])
        rouge_1_p_arr.append(scores[0]['rouge-1']['p'])
        rouge_1_f_arr.append(scores[0]['rouge-1']['f'])

        rouge_2_r_arr.append(scores[0]['rouge-2']['r'])
        rouge_2_p_arr.append(scores[0]['rouge-2']['p'])
        rouge_2_f_arr.append(scores[0]['rouge-2']['f'])

        rouge_l_r_arr.append(scores[0]['rouge-l']['r'])
        rouge_l_p_arr.append(scores[0]['rouge-l']['p'])
        rouge_l_f_arr.append(scores[0]['rouge-l']['f'])


# Calculating average scores

avg_rouge_1_r = sum(rouge_1_r_arr)/len(rouge_1_r_arr)
avg_rouge_1_p = sum(rouge_1_p_arr)/len(rouge_1_p_arr)
avg_rouge_1_f = sum(rouge_1_f_arr)/len(rouge_1_f_arr)

avg_rouge_2_r = sum(rouge_2_r_arr)/len(rouge_2_r_arr)
avg_rouge_2_p = sum(rouge_2_p_arr)/len(rouge_2_p_arr)
avg_rouge_2_f = sum(rouge_2_f_arr)/len(rouge_2_f_arr)

avg_rouge_l_r = sum(rouge_l_r_arr)/len(rouge_l_r_arr)
avg_rouge_l_p = sum(rouge_l_p_arr)/len(rouge_l_p_arr)
avg_rouge_l_f = sum(rouge_l_f_arr)/len(rouge_l_f_arr)

# avg_scores = {
#     'avg_rouge_1_r': avg_rouge_1_r,
#     'avg_rouge_1_p': avg_rouge_1_p,
#     'avg_rouge_1_f': avg_rouge_1_f,

#     'avg_rouge_2_r': avg_rouge_2_r,
#     'avg_rouge_2_p': avg_rouge_2_p,
#     'avg_rouge_2_f': avg_rouge_2_f,

#     'avg_rouge_l_r': avg_rouge_l_r,
#     'avg_rouge_l_p': avg_rouge_l_p,
#     'avg_rouge_l_f': avg_rouge_l_f,
# }

print('avg_rouge_1_r = ', avg_rouge_1_r)
print('avg_rouge_1_p = ', avg_rouge_1_p)
print('avg_rouge_1_f = ', avg_rouge_1_f)

print('avg_rouge_2_r = ', avg_rouge_2_r)
print('avg_rouge_2_p = ', avg_rouge_2_p)
print('avg_rouge_2_f = ', avg_rouge_2_f)

print('avg_rouge_l_r = ', avg_rouge_l_r)
print('avg_rouge_l_p = ', avg_rouge_l_p)
print('avg_rouge_l_f = ', avg_rouge_l_f)
