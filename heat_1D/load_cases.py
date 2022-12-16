
#%%
import pickle
import matplotlib.pyplot as plt

def load_case(case_dir, load_sol_data=False, load_prior_samples=False):
    samples = pickle.load(open(case_dir + '/samples.pkl', 'rb'))
    parameters = pickle.load(open(case_dir + '/parameters.pkl', 'rb'))
    if load_sol_data:
        x_exact = pickle.load(open(case_dir + '/x_exact.pkl', 'rb'))
        y_exact = pickle.load(open(case_dir + '/y_exact.pkl', 'rb'))
        data = pickle.load(open(case_dir + '/data.pkl', 'rb'))
        x_exact.geometry = parameters['x_exact_geometry']
        x_exact.is_par = parameters['x_exact_is_par']
        y_exact.geometry = parameters['y_exact_geometry']
        y_exact.is_par = parameters['y_exact_is_par']
        data.geometry = parameters['data_geometry']
        data.is_par = parameters['data_is_par']
    else:
        x_exact = None
        y_exact = None
        data = None
    if load_prior_samples:
        prior_samples = pickle.load(open(case_dir + '/prior_samples.pkl', 'rb'))
    else:
        prior_samples = None

    return prior_samples, samples, parameters, x_exact, y_exact, data

if __name__ == '__main__':
    prior_samples, samples, parameters, x_exact, y_exact, data = load_case('./data2_cont3/paper_case17')
    
    
    case_list = []
    case_list.append('./data2/paper_case2')
    case_list.append('./data2_cont2/paper_case11')
    case_list.append('./data2_cont2/paper_case12')
    case_list.append('./data2_cont2/paper_case13')
    case_list.append('./data2_cont2/paper_case14')
    case_list.append('./data2_cont2/paper_case16')
    case_list.append('./data2_cont3/paper_case17')
    
    load_sol_data_list = [False, False, False, False, False, False, True]
    load_prior_samples_list = [False, False, False, False, False, False, True]
    
    for i, case in enumerate(case_list):
        prior_samples, samples, parameters, x_exact, y_exact, data = load_case(case, load_sol_data_list[i], load_prior_samples_list[i])
        print(parameters)
        if load_sol_data_list[i]:
            plt.figure()
            x_exact.plot()
            y_exact.plot()
            data.plot()  

        if load_prior_samples_list[i]:
            plt.figure()
            for s in prior_samples:
                prior_samples.geometry.plot(s, is_par=True)    
    
        print('_'*80)
    