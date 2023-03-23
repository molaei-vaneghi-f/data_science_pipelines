import pandas as pd
import numpy as np
import os
from IPython.display import display, HTML
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns
import imageio

db = './data'
fert = pd.read_csv(os.path.join(db, 'gapminder_total_fertility.csv'), index_col=0)
life = pd.read_excel(os.path.join(db, 'gapminder_lifeexpectancy.xlsx'), index_col=0)
pop = pd.read_csv(os.path.join(db, 'population_total.csv'), index_col=0)

# data exploration
display(life)
life.shape
life.describe()
list(life.index) # life.index
list(life.columns) # life.columns
life.head(5) # df.head
life.tail(3)
life.sample(5)

#%% histogram of life data

ax = life[2015].hist(bins=20)
fig1 = ax.get_figure()
plt.title('gapminder life expectancy')
plt.xlabel('life expectancy in years')
plt.ylabel('number of samples')
fig1.savefig('../results/week1/life_exp_hist_final.png') #
plt.show()

#%% barplot

fert.sample(10)
fert.dropna()
fert00 = fert [['1800', '1950', '2015']]
fert11 = fert00.loc [['Germany', 'Martinique', 'Finland', 'India', 'Bulgaria', 'Kenya']]

ax = fert11.plot.bar(figsize=(10, 10))
fig1 = ax.get_figure()
plt.title('countries total fertility rate across years')
plt.xlabel('countries')
plt.ylabel('total fertility rate')
plt.show()
fig1.savefig('fertility_barplot.png') 

#%% 

# making all three data frames the same size
life = life.drop([2016], axis = 1, inplace = False) # CAUTION: inplace = True CHANGNGES THE ORIGINAL DF
# seeing if dfs have the same columns
life.columns == fert.columns
# life.columns == pop.columns

# change columns from str to int
fert.columns = fert.columns.astype(int)
pop.columns = pop.columns.astype(int)
# change index name
fert.index.name = 'country'
life.index.name = 'country'
pop.index.name = 'country'
# convert the table into long format, move the row index into a column
fert = fert.reset_index()
life = life.reset_index()
pop = pop.reset_index()
# conversion
fert = fert.melt(id_vars='country', var_name='year', value_name='fertility_rate')
life = life.melt(id_vars='country', var_name='year', value_name='life_expectancy')
pop = pop.melt(id_vars='country', var_name='year', value_name='population')

# concat
# combined_df = pd.concat([life, fert, pop])
# merging data frames
fert_pop = fert.merge(pop)
fert_pop.shape
fert_pop.sample(4)
fert_pop_life = fert_pop.merge(life)
fert_pop_life.shape
fert_pop_life.sample(4)

# dropping nan values
sns.heatmap(fert_pop_life.isna())
fert_pop_life.dropna(inplace=True)
sns.heatmap(fert_pop_life.isna())

#%% making scatterplots for all years
years = list(fert_pop_life['year'])

# for each year, make a scatterplot btw fertility and life exp
sns.set_style('whitegrid')

for year in range (1960, 2015): # (1960, 2015): # (before that, the data contains too many gaps)

    fert_pop_life_subset = fert_pop_life.loc[fert_pop_life['year'] == year]
    
    # define the symbol sizes based on the population values
    size = fert_pop_life_subset['population']
    
    sca = sns.scatterplot(data=fert_pop_life_subset, x='life_expectancy', y='fertility_rate', hue= 'country', size=size, sizes=(1, 200), alpha=0.6)
    # to remove the long legend composed of the population for each country
    plt.legend([])
   
    plt.title('life_expectancy vs fertility_rate in year ' + str(year))
    # plt.axis((xmin, xmax, ymin, ymax)) 
    plt.axis((20, 100, 0, 10))
    # plt.xlabel(' Year 1800')
    # plt.ylabel('Year 1950')
    fig = sca.get_figure()
    # fig.savefig('life_expectancy vs fertility_rate in year' + str(year), dpi = 400)
    fig.savefig('../results/life_expectancy_vs_fertility_rate_in_year_' + str(year), dpi = 400)
    plt.close()


# animating all scatterplots
images = []

for y in range(1960, 2015):
    # filename = 'lifeexp_{}.png'.format(i)
    filename = f'life_expectancy_vs_fertility_rate_in_year_{y}.png'
    # print(filename)
    images.append(imageio.imread('../results/' + filename))
    
fps = 10 # the lower the fps, the smoother the plot
imageio.mimsave(f'../results/life_expectancy_vs_fertility_rate_fps{fps}_last3.gif', images, fps=fps)



























