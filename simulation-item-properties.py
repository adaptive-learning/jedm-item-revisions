#!/usr/bin/python3

# "Towards Design-Loop Adaptivity: Identifying Items for Revision" paper
# Implementation of the experiment with simulated data (Section 5.1.2)

import sys
import math
import random
from collections import defaultdict
import numpy as np
import pylab as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set()

# scenarios represented input parameters of the simulation
# they represent different assumption about the learning situation
scenarios = {
    "basic": {
        "std": 1,
        "items": 47,
        "misfit_items": 3,
        "item_mean": 0,
        "delta": 0.03},
    "hard_items": {
        "std": 1,
        "items": 47,
        "misfit_items": 3,
        "item_mean": -1,
        "delta": 0.03},
    "no_learn": {
        "std": 1,
        "items": 47,
        "misfit_items": 3,
        "item_mean": 0,
        "delta": 0},
    "high_std": {
        "std": 2,
        "items": 47,
        "misfit_items": 3,
        "item_mean": 0,
        "delta": 0.03}
    }


def nans(x, y):
    out = np.empty((x, y))
    out.fill(np.nan)
    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Variation on the AFM model with easiness parameter for each item.
# The model is hardcoded with two knowledge components (KC) that are
# used to represent normal and misfitting items.
# - The first KC corresponds to "normal items".
# - The second KC corresponds to "missfitting items".
# All students answer all items in random order.
class SimulatedData:

    def __init__(self, scenario, students=2000):
        self.scenario = scenario
        self.students = students
        self.prob = nans(students, scenario["items"])
        self.answer = nans(students, scenario["items"])
        self.gen_params()
        self.gen_probabilities()
        self.gen_answers()

    def gen_params(self):
        std = self.scenario["std"]
        self.skill = np.random.normal(0, std, (2, self.students))
        self.easiness = np.array([np.random.normal(self.scenario["item_mean"], std)
                                  for _ in range(self.scenario["items"])])

    def gen_probabilities(self):
        for s in range(self.students):
            attempts0, attempts1 = 0, 0
            order = list(range(self.scenario["items"]))
            random.shuffle(order)
            for item in order:
                v = self.easiness[item]
                if item < self.scenario["misfit_items"]:
                    v += self.skill[0, s] + attempts0 * self.scenario["delta"]
                    attempts0 += 1
                else:
                    v += self.skill[1, s] + attempts1 * self.scenario["delta"]
                    attempts1 += 1
                self.prob[s, item] = sigmoid(v)

    def gen_answers(self):
        for s in range(self.students):
            for i in range(self.scenario["items"]):
                if random.random() < self.prob[s, i]:
                    self.answer[s, i] = 1
                else:
                    self.answer[s, i] = 0


# upper-lower discrimination index
def compute_discrimination_index(data):
    students, items = data.answer.shape
    student_sr = data.answer.mean(1)
    sorted_student_sr = sorted(student_sr)
    low_students = [s for s in range(students)
                    if student_sr[s] <= sorted_student_sr[len(student_sr)//3]]
    high_students = [s for s in range(students)
                     if student_sr[s] >= sorted_student_sr[2*len(student_sr)//3]]
    di = np.zeros(items)
    for i in range(items):
        di[i] = np.mean([data.answer[s, i] for s in high_students]) -\
            np.mean([data.answer[s, i] for s in low_students])
    return di


# point-biserial discrimination index
def compute_sr_cor(data):
    students, items = data.answer.shape
    user_sum = data.answer.sum(1)
    sr_cor = np.zeros(items)
    for i in range(items):
        x, y = [], []
        for s in range(students):
            answer = data.answer[s, i]
            x.append((user_sum[s] - answer) / items)
            y.append(answer)
        sr_cor[i] = pearsonr(x, y)[0]
    return sr_cor


# average item similarity
def compute_avg_sim(data):
    students, items = data.answer.shape
    cor = np.zeros((items, items))
    for i in range(items):
        for j in range(items):
            cor[i, j] = pearsonr(data.answer[:, i], data.answer[:, j])[0]
    avg_sim = cor.mean(0)
    return avg_sim


def check_agreement(disc_values, data):
    k = data.scenario["misfit_items"]
    k_val = sorted(disc_values)[k-1]
    c = 0
    for i in range(k):
        if disc_values[i] <= k_val:
            c += 1
    return c


def analyze_discrimination(scenario="basic", rep=20):
    indicies = [("di", compute_discrimination_index),
                ("avg_sim", compute_avg_sim),
                ("sr_cor", compute_sr_cor)]
    student_counts = [20, 50, 100, 200, 400, 600]
    results = {i: defaultdict(list) for i, _ in indicies}
    k = scenarios[scenario]["misfit_items"]
    plt.figure(figsize=(6, 6))
    for students in student_counts:
        for _ in range(rep):
            data = SimulatedData(scenarios[scenario], students)
            for desc, fun in indicies:
                disc_values = fun(data)
                agreement = check_agreement(disc_values, data)
                print(students, desc, agreement)
                results[desc][students].append(agreement / k)
    for ind in results:
        plt.plot(student_counts,
                 [np.mean(results[ind][s]) for s in student_counts],
                 label=ind)
    plt.title(scenario)
    plt.ylim((0, 1.1))
    plt.legend()
    plt.xlabel("answers per item")
    plt.ylabel("average agreement with the ground truth")
    plt.savefig(f"sim-discr-{scenario}.svg")
    plt.show()


if len(sys.argv) < 2:
    scenario = "basic"
else:
    scenario = sys.argv[1]

analyze_discrimination(scenario)
