
#%%
import pickle
import matplotlib.pyplot as plt

def load_case(case_dir, load_sol_data=False):
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
    return samples, parameters, x_exact, y_exact, data


samples, parameters, x_exact, y_exact, data = load_case('./data2_cont3/paper_case17')

#%%
case_list = []
case_list.append('./data2/paper_case2')
case_list.append('./data2_cont2/paper_case11')
case_list.append('./data2_cont2/paper_case12')
case_list.append('./data2_cont2/paper_case13')
case_list.append('./data2_cont2/paper_case14')
case_list.append('./data2_cont2/paper_case16')
case_list.append('./data2_cont3/paper_case17')

load_sol_data_list = [False, False, False, False, False, False, True]

for i, case in enumerate(case_list):
    samples, parameters, x_exact, y_exact, data = load_case(case, load_sol_data_list[i])
    print(parameters)
    if load_sol_data_list[i]:
        plt.figure()
        x_exact.plot()
        y_exact.plot()
        data.plot()        

    print('_'*80)
