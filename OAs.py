import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)


@ti.func
def clip(pop: ti.template(), lb: ti.template(), ub: ti.template()):

    search_num, dim = pop.shape
    for i, j in ti.ndrange(search_num, dim):
        if pop[i, j] > ub[j]:
            pop[i, j] = ub[j]
        elif pop[i, j] < lb[j]:
            pop[i, j] = lb[j]


@ti.func
def clip_only(trial: ti.template(), lb: ti.template(), ub: ti.template()):

    dim = trial.shape[0]
    for j in range(dim):
        if trial[j] > ub[j]:
            trial[j] = ub[j]
        elif trial[j] < lb[j]:
            trial[j] = lb[j]


@ti.func
def f1(fit: ti.template(), pop: ti.template()):

    search_num, dim = pop.shape
    for i in range(search_num):
        cur = 0.0
        for j in range(dim):
            cur += ti.pow(a=pop[i, j], b=2)

        fit[i] = cur


@ti.func
def f1_only(trial: ti.template()) -> ti.float32:

    dim = trial.shape[0]
    res = 0.0
    for j in range(dim):
        res += ti.pow(a=trial[j], b=2)

    return res


@ti.func
def find_min(fit: ti.template()) -> ti.i32:

    search_num = fit.shape[0]
    min_fit = fit[0]
    min_pos = 0
    for _ in ti.ndrange(1):
        for i in ti.ndrange(search_num):
            if min_fit < fit[i]:
                min_fit = fit[i]
                min_pos = i
    return min_pos


@ti.func
def rand_int(low: ti.i32, high: ti.i32) -> ti.i32:

    r = ti.random(float)
    res = r * (high - low) + low

    return ti.round(res, dtype=ti.i32)


@ti.func
def copy_pop_to_field(pop: ti.template(), trial: ti.template(), ind: ti.i32):

    search_num, dim = pop.shape
    for j in range(dim):
        trial[j] = pop[ind, j]


@ti.func
def copy_field_to_pop(pop: ti.template(), trial: ti.template(), ind: ti.i32):

    search_num, dim = pop.shape
    for j in range(dim):
        pop[ind, j] = trial[j]


@ti.func
def copy_field_a_to_b(a: ti.template(), b: ti.template()):
    dim = a.shape[0]
    for j in range(dim):
        b[j] = a[j]


@ti.func
def de_crossover(pop: ti.template(), trial: ti.template(), a: ti.i32, b: ti.i32, c: ti.i32, ind: ti.i32):

    search_num, dim = pop.shape
    CR = 0.5
    para_F = 0.7
    for k in range(dim):
        r = ti.random(float)
        if r < CR or k == dim - 1:
            trial[k] = pop[c, k] + para_F * (pop[a, k] - pop[b, k])


@ti.func
def de_loop(pop: ti.template(), all_best: ti.float32, fit: ti.template(), trial: ti.template(), lb: ti.template(), ub: ti.template()) -> ti.float32:

    search_num, dim = pop.shape
    for i in range(search_num):

        copy_pop_to_field(pop=pop, trial=trial, ind=i)

        a = rand_int(low=0, high=search_num)
        while a == i:
            a = rand_int(low=0, high=search_num)

        b = rand_int(low=0, high=search_num)
        while b == i or a == b:
            b = rand_int(low=0, high=search_num)

        c = rand_int(low=0, high=search_num)
        while c == i or c == a or c == b:
            c = rand_int(low=0, high=search_num)

        de_crossover(pop=pop, trial=trial, a=a, b=b, c=c, ind=i)
        clip_only(trial=trial, lb=lb, ub=ub)
        next_fit = f1_only(trial=trial)
        if next_fit < fit[i]:
            copy_field_to_pop(pop=pop, trial=trial, ind=i)
            fit[i] = next_fit
            if next_fit < all_best:
                all_best = next_fit
                copy_field_a_to_b(a=trial, b=best_pop)

    return all_best


@ti.kernel
def DE(pop: ti.template(), max_iter: ti.i32, lb: ti.template(), ub: ti.template(), fit: ti.template(), best_fit: ti.template(), best_pop: ti.template(), trial: ti.template()):

    f1(fit=fit, pop=pop)
    min_pos = find_min(fit=fit)
    all_best = fit[min_pos]
    best_fit[0] = all_best

    for _ in range(1):
        for cur_iter in range(1, max_iter + 1):

            # if cur_iter % 50000 == 0:
            #     print(cur_iter)

            all_best = de_loop(pop=pop, fit=fit, all_best=all_best, trial=trial, lb=lb, ub=ub)
            best_fit[cur_iter] = all_best


import time

search_num = 30
dim = 30
max_iter=300000

start = time.time()
_lb = np.ones(dim).astype(np.int32) * (-100)
lb = ti.field(ti.i32, shape=dim)
lb.from_numpy(_lb)

_ub = np.ones(dim).astype(np.int32) * 100
ub = ti.field(ti.i32, shape=dim)
ub.from_numpy(_ub)

pop = ti.field(ti.float32, shape=(search_num, dim))
pop.from_numpy((np.random.random((search_num, dim)) * (_ub - _lb) + _lb).astype(np.float32))

fit = ti.field(ti.float32, shape=(search_num, ))
best_fit = ti.field(ti.float32, shape=(max_iter, ))
best_pop = ti.field(ti.float32, shape=(search_num, ))

trial = ti.field(ti.float32, shape=(search_num, ))

DE(pop=pop, max_iter=max_iter, lb=lb, ub=ub, fit=fit, best_fit=best_fit, best_pop=best_pop, trial=trial)

res = best_fit.to_numpy()
print(res[-1])

print(f'run time: {time.time() - start}')
print('done')
