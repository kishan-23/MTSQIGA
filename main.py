from quantumGA import QuantumGA
from preproc import Preprocessor
from rouge import Rouge

from os import listdir, mkdir
from os.path import (
    join,
    isdir,
    exists
)


file_list = [
    '001.txt',
    '002.txt',
    '003.txt',
    '004.txt',
    '005.txt',
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

for file_last_name in file_list:
    file_name = r'.\data\bbc_news_data\article\business'+'\\'+file_last_name
    file = open(file_name)
    doc_title = file.readline()
    print()
    print("doc_title = ", doc_title)
    file.readline()
    text = file.read()
    # print("text = ", text)
    file.close()

    file_name = r'.\data\bbc_news_data\summaries\business'+'\\'+file_last_name
    file = open(file_name)
    ref_summary = file.read()
    # print("ref_summary = ", ref_summary)
    file.close()

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
    designed_sizes = [int(preProc.sentencesNum/4)]

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
            mating_pool = quantumGA.rouletteWheel(
                quantum_pop, pop_size)
            # q_offsprings = quantumGA.twoPointCrossover(mating_pool)
            # q_offsprings = quantumGA.singlePointCrossover(mating_pool)
            q_offsprings = quantumGA.uniformCrossover(mating_pool)
            quantumGA.cusMutation(q_offsprings)
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
