#For visualizing data frames

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def make_corr_matrix(df, file_name, title):
    ''' This function makes a correlation matrix with shades of red.  Floats and integers are formatted
        to fit in the matrix.
        df: data frame
        file_name:  file name taken in string format
		title: title of the matrix in string format
    '''
    correlation = df.corr().round(2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlation, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    setting = "g"
    plt.figure()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation, mask=mask, annot=True,cmap='Reds', fmt = setting, linewidths=.5, ax=ax)
    plt.title("Correlations Between " + title, fontsize=20)
    plt.savefig(file_name)
	
def make_reg_plt (x_label,y_label,df,file_name):
    '''This function makes a simple regression plot between two fields
        x_label: field for the x-axis in string format
        y_lable: field for the y-axis in string format
        df: data frame
        file_name: file name taken in string format
    '''

    sns.regplot(x=x_label, y=y_label, data=df)
    plt.suptitle("Correlation Between " + x_label + " and " + y_label)
    plt.show()
    plt.savefig(file_name)

def scttr_matrix(df,file_name,title):
    '''This function takes a scatter matrix.
        df: data frame
        file_name: file name taken in string format
        title: title of the scatter matrix
    '''

    plt.figure()
    p = sns.pairplot(data=df, dropna=True, palette="husl",diag_kind = "kde")

    #mask upper triangle

    for i,j in zip(*np.triu_indices_from(p.axes,1)):
        p.axes[i, j].set_visible(False)

    p.fig.text(0.33, 1.02, "A High-Level View of Relationships Between "+ title, fontsize=20)
    plt.show()
    plt.savefig(file_name)



