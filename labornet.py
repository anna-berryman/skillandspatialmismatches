'''Network Labor Model
@rmaria del rio-chanona
'''
import copy
import numpy as np

############################
# Code for running the agent-based model (solving the analytical equations)
# for running the agent-based model without approximations see code in Julia
############################

def matching_probability(sij, vj):
    ''' probability of an application sent to occupation j is successful
    '''
    # get supply of workers for j
    sj = np.sum(sij, axis=0)
    sj_inv = np.array([1./s for s in sj])
    # get labor market tightness of an occupation
    θj = np.multiply(vj, sj_inv)
    with np.errstate(divide='ignore', invalid='ignore'):
        θj = np.multiply(vj, sj_inv)
        θj_inv = np.array([1./θ for θ in θj])
    θj_inv[np.isnan(θj_inv)] = 0
    θj[np.isnan(θj)] = 0
    # get probability of application to occupation j being succesfull
    pj = np.multiply(θj, 1 - np.exp(-θj_inv))
    return pj


def fire_and_hire_workers(parameters, variables, d_dagger_t,\
    A, matching_probability):
    '''
    Function that updates the number of employed and unemployed workers and
    job vacancies in each occupation. The update considers the spontanous
    separations and opening of vacancies, as well as the directed separations
    and openings and flow of workers.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    Args:
        parameters(list): δ_u, δ_v, γ_u, γ_v (floats)
        variables (list): with employment, u, v (np arrays) with employment
            unemployment and vacancies for each occupation
        d_dagger_t(np arrays):target demand for each occupation
        A(np arrays): adjacency matrix
        matching_probability(function): probability of app in j being succesfull
    '''
    assert(len(parameters) == 4), "wrong number of parameters"
    δ_u, δ_v, γ_u, γ_v = parameters

    employment, u, v = variables
    n = len(employment)
    # spontanous separations and job opening
    separated_workers = δ_u*employment
    opened_vacancies = δ_v*employment
    # state dependent separations and job openings (directed effort)
    # compute difference between realized and target demand
    Δ_demand = employment + v + - d_dagger_t
    # get extra separations and openings, bounded by employment of occupation
    g_u = np.minimum(employment, γ_u * np.maximum(np.zeros(n), Δ_demand))
    g_v = np.minimum(employment, γ_v * np.maximum(np.zeros(n), -Δ_demand))
    # get number of separations and job openings
    separated_workers = separated_workers + (1 - δ_u)*g_u
    opened_vacancies = opened_vacancies + (1 - δ_v)*g_v

    # Search and matching
    Av = np.multiply(v, A)
    # get matrix q_ij, probability of applying from i to j
    Q = Av / np.sum(Av, axis=1,keepdims=1)
    # expected number of job applications from i to j
    sij = np.multiply(u[:, None], Q)
    # expected probability of application being succesfull
    pj = matching_probability(sij, v)
    # expected flow of workers
    F = np.multiply(sij, pj)#sij .* pj


    # getting hired workers and transitioning workers
    hired_workers = np.sum(F, axis=0)
    exported_workers = np.sum(F, axis=1)

    # print("vacancies ", v)
    # print("expected applications ", sij.sum(axis=0))
    # print("matches ", hired_workers)
    # print("outflow unemployment", hired_workers.sum())
    # print("separations ", separated_workers.sum())
    # print("open vac", opened_vacancies.sum())
    # print("sep ", separated_workers)
    # print("exp ", exported_workers)
    # print("u ", u)

    # check everything okay with code
    assert(min(hired_workers) >= 0)
    assert(min(exported_workers) >= 0)

    # print("unemployed = ",u.sum())
    # print("vacancies = ",v.sum())
    # print("probabilities = ",pj.sum())
    # print("hired workers = ",hired_workers.sum())
    # print("exported_workers = ",exported_workers.sum())


    #Update e, w and v
    u += separated_workers - exported_workers
    employment += hired_workers - separated_workers
    v += opened_vacancies - hired_workers


    # make sure variables are postive (accounting for floating point error)
    # assert(minimum(v) >= 0.0 || isapprox(v, zeros(n); atol=1e-15, rtol=0))
    # assert(minimum(u) >= 0.0 || isapprox(u, zeros(n); atol=1e-15, rtol=0))
    # assert(minimum(employment) >= 0.0 || isapprox(employment,
    #         zeros(n); atol=1e-15, rtol=0))
    # now that we know variables are non negative, correct floating point error
    v = np.maximum(1e-15 * np.ones(n), v)
    u = np.maximum(1e-15 * np.ones(n), u)
    employment = np.maximum(1e-15 * np.ones(n), employment)

    variables = [employment, u, v]
    changes = [separated_workers, exported_workers, opened_vacancies, hired_workers]
    return variables, changes



def run_numerical_solution(fire_and_hire_workers, t_sim, parameters,\
    variables_0, target_demand_function, D_0, scn_dict, scn,\
    t_shock, shock_dur, matching, A_matrix, τ):
    """Iterates the firing and hiring process for n_steady steps. Then runs the simulation with the
    restructured employment for n_sim time steps.
    returns unemployment rate, w, v, e and W lists.
    Args:
    fire_and_hire_workers(function): function that updates w, v, e and W.
    A_matrix(array): adjacency matrix of the network
    t_sin(int): number of times the simulation is run (after steady state)
    delta_u, delta_v, gamma_u, gamma_v(Float64): parameters of network
    employment_0(array(n_occ, 1)): number of employed workers at initial time.
    """
    employment_0, unemployment_0, vacancies_0 = variables_0
    assert(len(employment_0) == len(vacancies_0) == len(unemployment_0))
    n_occ = len(employment_0)
    initial_variables = []
    # setting initial conditions
    employment = copy.deepcopy(employment_0)
    unemployment= copy.deepcopy(unemployment_0)
    vacancies = copy.deepcopy(vacancies_0)
    new_variables = [employment, unemployment, vacancies]
    # defining arrays where information is stored
    E = np.zeros([t_sim, n_occ])
    U = np.zeros([t_sim, n_occ])
    V = np.zeros([t_sim, n_occ])
    D = np.zeros([t_sim, n_occ])
    D[0, :] = D_0

    # recording initial conditions
    E[0, :] = employment_0
    U[0, :] = unemployment_0
    V[0, :] = vacancies_0
    Variables= [E, U, V]
    U_all = np.zeros([t_sim, t_sim, n_occ])
    V_all = np.zeros([t_sim, t_sim, n_occ])

    for t in range(1,t_sim):
        if t%50==0:
            print(t)
    #for t in range(t_sim-1):
        # compute target demand for given time step
        d_dagger_t = target_demand_function(t, D_0, scn_dict, scn, \
                                            t_shock, shock_dur)
        D[t, :] = d_dagger_t
        # update main variables and get the number of separations
        new_variables, changes = fire_and_hire_workers(parameters, \
            new_variables, d_dagger_t, A_matrix, matching)
        separated, exported, opened_vacancies, filled = changes
        # the number of separations = unemployed workers with 1 t.s. of unemp
        U_all[t, 0, :] = separated
        # the number of unfilled vacancies = vacancies with 1 t.s. of being open
        V_all[t, 0, :] = opened_vacancies
        # fill in expected number of unemployed workers with given job spell
        # note that max job spell is time of simulation so far
        # fill in for more than 1 time step
        for n in range(1,t+1):
            # job spell is those of previous job spell - the ones hired
            # note Variables[1] = unemployed list
            U_all[t, n, :] = U_all[t - 1, n - 1, :] * (1 - exported/(Variables[1][t - 1, :]))
            # vacancy open spell is those previous open spell - the ones filled
            # note Variables[2] = vacancy list
            V_all[t, n, :] = V_all[t - 1, n - 1, :] * (1 - filled/(Variables[2][t - 1, :]))
        # store information in arrays
        Variables[0][t, :] = new_variables[0]
        Variables[1][t, :] = new_variables[1]
        Variables[2][t, :] = new_variables[2]


    return Variables, U_all, D, V_all

