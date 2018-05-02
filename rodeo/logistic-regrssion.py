# http://blog.yhat.com/posts/logistic-regression-python-rodeo.html

# hasthags indicate notes about code; the code below imports a few packages we will need for this analysis
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# read the data in
df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
print df.head()

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print df.columns

# summarize the data
print df.describe()

# take a look at the standard deviation of each column
print df.std()

# frequency table cutting prestige and whether or not someone was admitted
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

# plot all of the columns
df.hist()
pl.show()

# dummify rank
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print dummy_ranks.head()

# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print data.head()

# manually add the intercept
data['intercept'] = 1.0


train_cols = data.columns[1:]

logit = sm.Logit(data['admit'], data[train_cols])

# fit the model
result = logit.fit()

# cool enough to deserve it's own comment
print result.summary()

# look at the confidence interval of each coeffecient
print result.conf_int()

# odds ratios only
print np.exp(result.params)

# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)

# instead of generating all possible values of GRE and GPA, we're going
# to use an evenly spaced range of 10 values from the min to the max
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
print gres
gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print gpas

# define the cartesian function
def cartesian(arrays, out=None):
  arrays = [np.asarray(x) for x in arrays]
  dtype = arrays[0].dtype

  n = np.prod([x.size for x in arrays])
  if out is None:
      out = np.zeros([n, len(arrays)], dtype=dtype)

  m = n / arrays[0].size
  out[:,0] = np.repeat(arrays[0], m)
  if arrays[1:]:
      cartesian(arrays[1:], out=out[0:m,1:])
      for j in xrange(1, arrays[0].size):
          out[j*m:(j+1)*m,1:] = out[0:m,1:]
  return out


# enumerate all possibilities
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))

# recreate the dummy variables
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']

# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

# make predictions on the enumerated dataset
combos['admit_pred'] = result.predict(combos[train_cols])

print combos.head()



def isolate_and_plot(variable):
  # isolate gre and class rank
  grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'prestige'],
              aggfunc=np.mean)
  # make a plot
  colors = 'rbgyrbgy'
  for col in combos.prestige.unique():
      plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
      pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'], color=colors[int(col)])

  pl.xlabel(variable)
  pl.ylabel("P(admit=1)")
  pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
  pl.title("Prob(admit=1) isolating " + variable + " and prestige")
  pl.show()

isolate_and_plot('gre')
isolate_and_plot('gpa')
