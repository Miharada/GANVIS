
from flask import Flask, render_template, request
import pandas as pd
import pygad
import numpy as np
import GenAlg as GA
app = Flask(__name__)

data=0

@app.route('/')
@app.route('/home')
def home_page():
    init_gen = 25
    init_paretmating = 2
    init_solution = 0
    init_genes = 0
    init_low = -4
    init_high = 4
    init_parentselect = 0
    init_keepparent = -1
    init_crosovertype = 0
    init_crosoverprob = None
    init_mutype = 0
    init_mutprob = None

    return render_template('base.html', 
                            generation=init_gen, 
                            parentMating=init_paretmating, 
                            solution=init_solution, 
                            genes=init_genes,
                            low=init_low,
                            high=init_high,
                            keepParent=init_keepparent,
                            crosprob=init_crosoverprob,
                            mutProb=init_mutprob)

#https://www.py4u.net/discuss/278063
@app.route('/', methods=['POST'])
def testing_data():

    parentSelect = ["sss", "rws", "sus", "rank", "random", "tournament"]
    crosovertype = ["single_point", "two_points", "uniform", "scattered"]
    mutationType = ["random", "swap", "inversion", "scramble", "adaptive"]

    generation = int(request.form['InputGeneration'])
    parentMating = int(request.form['InputParentsMating'])
    solution = int(request.form['InputSol'])
    genes = int(request.form['InputGenes'])
    init_low = float(request.form['initLow'])
    init_high = float(request.form['initHigh'])
    parentidx = int(request.form['parentSelection'])
    keepParent = int(request.form['keepParent'])
    crossoveridx = int(request.form['crossoverType'])
    crosovProb = (request.form['crossoverProb'])
    mutTypeidx = int(request.form['mutType'])
    mutProb = (request.form['mutProb'])


    if crosovProb == "":
        crosovProb = None
    else:
        crosovProb = float(crosovProb)
    if mutProb == "":
        mutProb = None
    else:
        mutProb = float(mutProb)

    ga_instance = pygad.GA(num_generations=generation,
                            num_parents_mating=parentMating,
                            sol_per_pop=solution,
                            num_genes=genes,
                            init_range_low=init_low,
                            init_range_high=init_high,
                            parent_selection_type=parentSelect[parentidx],
                            keep_parents=keepParent,
                            crossover_type=crosovertype[crossoveridx],
                            crossover_probability=crosovProb,
                            mutation_type=mutationType[mutTypeidx],
                            mutation_probability=mutProb,
                            fitness_func = fitness_func
                            )
    ga_instance.run()
    # ga_instance.plot_fitness()
    print(crosovProb, mutProb)
    values = ga_instance.best_solutions_fitness
    labels = [i for i in range(len(values))]

    return render_template('home.html', 
                            generation=generation, 
                            parentMating=parentMating, 
                            solution=solution, 
                            genes=genes,
                            low=init_low,
                            high=init_high,
                            keepParent=keepParent,
                            crosprob=crosovProb,
                            mutProb=mutProb,
                            labels=labels,
                            values=values)
#https://pythonbasics.org/flask-upload-file/
@app.route('/test', methods=['POST'])
def upload_dataset():
    data = request.form['InputFile']
    print(type(data))
    return render_template('home.html')





function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.


def fitness_func(solution, solution_idx):

    output = np.sum(solution*function_inputs)
    fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness


target_chromosome = ""
def fitness_func_img(solution, solution_idx):
    fitness = np.sum(np.abs(target_chromosome-solution))

    fitness = np.sum(target_chromosome) - fitness
    return fitness