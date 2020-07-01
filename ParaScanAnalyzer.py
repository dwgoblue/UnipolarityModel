import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def Analysis(file_path, n):
    file = pd.read_pickle(file_path)
    tot_res, u_res, m_res, t_res = [], [], [], []
    col = []
    row = []
    for i, f in enumerate(file.items()):
        d = f[1]
        col.append(np.round(np.log10(d['kd']), 2))
        row.append(np.round(np.log10(d['ks']), 2))
        tot_diff = (d['Data'][0]+d['Data'][1])/(d['Data'][2]+d['Data'][3])
        u_diff = (d['Data'][0])/(d['Data'][2])
        m_diff = (d['Data'][1])/(d['Data'][3])
        steps = d['Data'][-1]
        tot_res.append(tot_diff)
        u_res.append(u_diff)
        m_res.append(m_diff)
        t_res.append(steps)

    matrix_col = np.array(col).reshape(n, n)
    matrix_row = np.array(row).reshape(n, n)
    
    matrix_res = np.array(tot_res).reshape(n, n)
    df1 = pd.DataFrame(matrix_res, index=np.unique(matrix_row), 
                      columns=np.unique(matrix_col))
    
    matrix_res = np.array(u_res).reshape(n, n)
    df2 = pd.DataFrame(matrix_res, index=np.unique(matrix_row), 
                      columns=np.unique(matrix_col))
    
    matrix_res = np.array(m_res).reshape(n, n)
    df3 = pd.DataFrame(matrix_res, index=np.unique(matrix_row), 
                      columns=np.unique(matrix_col))
    
    matrix_res = np.array(t_res).reshape(n, n)
    df4 = pd.DataFrame(matrix_res, index=np.unique(matrix_row), 
                      columns=np.unique(matrix_col))
    
    return (df1, df2, df3, df4)
    
    
if __name__ == '__main__':
    # Change the path.
    file_path = r'data/[20200630][scan][kd][ks][10+5mss]Morpho_result.pickle'
    dfs = Analysis(file_path, n)
    matplotlib.rcParams.update({'font.size': 22})
    title = ['total', 'diffuse', 'membrane-bound', 'steps']
    
    for df, tit in zip(dfs, title):
        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111)
        ax = sns.heatmap(df, ax=ax, annot=True, linewidths=.5)
        ax.set_ylim(10, 0)
        ax.set_xlabel('kd (log10)', fontsize=28)
        ax.set_ylabel('ks (log10)', fontsize=28)
        ax.set_title('Ratio of {} signals from two poles'.format(tit), fontsize=22)
        plt.show()