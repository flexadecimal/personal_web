---
layout: post_toc
title:  "Racing data science with Python, Pandas, and NumPy"
slug: "racing_data_science"
date:   2020-05-15
categories: [python, data_science, cars, math]
author: David Kotaev
preamble: |-
  Building and racing cars can be an incredibly expensive hobby. Why not use some of our favorite data science tools to produce quantitative information to inform shopping decisions?

  For a data science class's group project, I elected to use dyno charts as the input data set. This writeup is an abridged version of the full project - see the [full source code](https://github.com/flexadecimal/cs301-project/blob/master/deliverables.ipynb) for the [Jupyter](https://jupyter.org/) notebook for more details.

  If you are a business, automotive or otherwise, and would like to contact me with professional inquiries, please see the [Contact page](../contact).
---
## What's a "dyno chart", anyways?
A dynamometer is a machine used to measure the power output of a car over its RPM range - essentially a hamster wheel for cars.
<figure>
  <img src="{{site.url}}{{site.baseurl}}/static/images/dyno.jpg" alt="Dynamometer"/>
  <figcaption>A dynamometer, commonly known as a "dyno".</figcaption>
</figure>
A _dyno chart_ is the output of this machine, and gives you horsepower and torque as a function of RPM (revolutions per minute, itself a measurement of time).
<figure>
  <img src="{{site.url}}{{site.baseurl}}/static/images/dynochart.jpg" alt="Dynamometer"/>
  <figcaption>A dyno chart.</figcaption>
</figure>
The dyno charts in this project were sourced from [Dymomite!](http://www.primaryboost.com/3s/dynocharts/), which is a collection of user-submitted dynocharts from forums for the Mitsubishi 3000 aka GTO, imported into the U.S. as the Dodge Stealth.
<figure>
  <img src="{{site.url}}{{site.baseurl}}/static/images/mitsu_3k.jpg" alt="Mitsubishi 3000"/>
  <figcaption>Mitsubishi 3000.</figcaption>
</figure>

## What's the data look like?
A dyno trial consists of:
  - run data: (x, y) points representing the dyno chart horsepower and torque functions, e.g. ```[{'x': 2510, 'y': 53}, {'x': 2921, 'y': 107}]```
  - a mods dictionary: describes the modifications the car has, with mods as keys and values as specification, e.g. ```{'turbo_class': 'td04', 'turbo_name': '17g'}```
  - associated metadata, e.g. forum username

The trials were split into horsepower and torque trials. Horsepower and torque are linearly related:
{%katex display%}\text{HP} = \frac{\text{RPM}*\text{T}}{5252}{%endkatex%}
...but are useful for answering different questions.

See the [GitHub repository](https://github.com/flexadecimal/cs301-project) for the JSON files used to import into Pandas.

```python
trials = pd.read_json('./dynomite_trials.json')
mods = pd.read_json('./dynomite_mods.json')
# replace mod numbers with human readable mods
mod_idx_name_dict = {
        '1':'boost', '2':'fuel',
        '3':'displacement', '4':'turbo_class',
        '5':'turbo_name', '7':'injection', '8':'nitrous',
        '9':'correction', '10':'heads',
        '11':'max_hp', '12':'max_tq'
};
# replace with human readable mods
readable_mods = [{mod_idx_name_dict[k]:v for k, v in r['mods'].items()} for idx, r in trials.iterrows()]
trials['mods'] = readable_mods
# split into hp/torque trials - make copies because we will add cols later
hp_trials = trials.copy().loc[trials['data_type_id'] == 1]
torque_trials = trials.copy().loc[trials['data_type_id'] == 2]
trials.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>display_info</th>
      <th>name</th>
      <th>displayName</th>
      <th>owner_id</th>
      <th>car_id</th>
      <th>run_id</th>
      <th>data_type_id</th>
      <th>mods</th>
      <th>run_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>{'username': '11secondFWD'}</td>
      <td>11secondFWD HP 9 1</td>
      <td>11secondFWD HP (17g)</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>{'turbo_class': 'td04', 'turbo_name': '17g'}</td>
      <td>[{'x': 2510, 'y': 53}, {'x': 2921, 'y': 107}, ...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>{'username': '11secondFWD'}</td>
      <td>11secondFWD TQ 9 1</td>
      <td>11secondFWD TQ (17g)</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>{'turbo_class': 'td04', 'turbo_name': '17g'}</td>
      <td>[{'x': 2514, 'y': 118}, {'x': 2619, 'y': 143},...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>{'username': '2root4u'}</td>
      <td>2root4u HP 18 1</td>
      <td>2root4u HP (gt368)</td>
      <td>18</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>{'boost': '25', 'turbo_class': 'td04', 'turbo_...</td>
      <td>[{'x': 3564, 'y': 210}, {'x': 3942, 'y': 262},...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>{'username': '3sx'}</td>
      <td>3sx HP 6 1</td>
      <td>3sx HP (GT35)</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>{'turbo_class': 'td05', 'turbo_name': 'GT35', ...</td>
      <td>[{'x': 4200, 'y': 200}, {'x': 4800, 'y': 370},...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>{'username': '97vr4'}</td>
      <td>97vr4 HP 15 1</td>
      <td>97vr4 HP (19t)</td>
      <td>15</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>{'boost': '25', 'turbo_class': 'td04', 'turbo_...</td>
      <td>[{'x': 3457, 'y': 201}, {'x': 3686, 'y': 244},...</td>
    </tr>
  </tbody>
</table>
### Supporting code - statistical functions and mod filtering
Definitions have been ommitted for brevity.
```python
# regular 5-figure stats summary: outliers, quartiles, min, max
def q_summary(numbers)
# 5-figure stats summary takes Pandas DataFrames as input and returns dictionary of keys min, max, outliers, quartiles and Dataframe row values
def q_summary_dataframe(df, col)
# pretty print 5-figure df summary
def print_df_summary(summary)

# filter trials by acceptable mods. acceptable mods is specified by a dictionary of modname:[acceptable values], e.g.
# stockengine_spec = {
#     'injection':[None],
#     'nitrous':[None],
#     'heads':['stock', None]
# }
def filter_by_mods(trials, mods_spec)
````

## Integrals and derivatives as aggregate functions
The run data in our data set is a set of (x, y) points representing the trial's horsepower or torque curve. The usual five-figure statistical summary of min/max, quartiles, and outliers is usually done on a one dimensional set of numbers, so we need some functions to meaningfully flatten the set of points to a single number.

### Integrals - the sum of it all
Enter... the amazing _**bounded integral!**_
<figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Integral_example.svg/500px-Integral_example.svg.png" alt="Example of a bounded integral"/>
  <figcaption>A bounded integral.</figcaption>
</figure>
If you've taken a calculus class, this will certainly look familiar. In short, the integral from point _a_ to point _b_ gives you the "area under the graph", or more generally, the total accumulation of your value with respect to your x-axis variable.

For the horsepower and torque functions, the points _a_ and _b_ denote a range of RPMs, itself a function of time.
- For horsepower, the integral is the total power made over the RPM range. Power, from a physics perspective, is the amount of work done per unit time.
- For torque, the integral is the total force made over the RPM range. Force is more colloquially thought of as the pull and push - the thing that pushes you back in your seat during acceleration!

The following function uses the NumPy method [```trapz()```](https://numpy.org/doc/1.18/reference/generated/numpy.trapz.html) to approximate the integral from the set of (x, y) run data points. Unlike your calculus class, real world data is often quantized and given as a discrete collection like a list of (x, y) points.

Unfortunately, the trial run data lists are not all the same size - different dynos record with different resolutions, hence different numbers of points. Rather than choose RPM ranges for integration randomly, I used statistical summary to find the median start and end points.

```python
# print out a 5-figure stats summary and return the integration bounds (median start and end rpm)
def summarize_trials_bounds(trials, name):
    # only interested in 'x' part, i.e. rpm
    start_rpms = [run[0]['x'] for run in trials]
    end_rpms = [run[-1]['x'] for run in trials]
    start_summary = q_summary(start_rpms)
    end_summary = q_summary(end_rpms)
    print('{} start RPM summary: {}'.format(name, start_summary))
    print('{} end RPM summary: {}'.format(name, end_summary))
    start = start_summary['q2']
    end = end_summary['q2']
    return (start, end)

# integrate run data [(x,y)...], filtering out values outside the [a,b] interval
def definite_discrete_integral(run, a, b):
    # filter out any data points with rpm less than a, greater than b
    filtered = [p for p in run if p['x'] >= a and p['x'] <= b]
    # discrete integral using trapezoidal rule
    xs = [p['x'] for p in filtered]
    ys = [p['y'] for p in filtered]
    return np.trapz(ys, x=xs)
```
Strictly speaking, the gradient is a _list_ of values, so not an aggregate function like the integral - but in this application we take the ```max()``` of this list to render a single value.

### Derivatives and gradients - how curvy that line is, basically
<figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Tangent_to_a_curve.svg/500px-Tangent_to_a_curve.svg.png" alt="Derivative"/>
  <figcaption>The derivative at a point, a.k.a the "tangent" line.</figcaption>
</figure>
Technically speaking, the derivative is the instantaneous rate of change. The commonly-used physics example is that speed is the derivative of position, and acceleration is the derivative of speed. 

In our application, the derivative at any point of a horsepower curve is the the measure of how much the power delivery is changing at that instant RPM.

The gradient is simply a collection of derivatives - rather than taking a single derivative at a point, the gradient is the derivative at _every_ point. Gradients are most commonly introduced and used in multivariate calculus, but when used with our univariate horsepower curve, gives us the "bumps" in power delivery. **This will be used later!**

The following function calculates the gradient for a run, i.e a list of the discrete derivatives between each point.
```python
def discrete_gradient(run_data):
    run_x = [p['x'] for p in run_data]
    run_y = [p['y'] for p in run_data]
    return list(np.gradient(run_y, run_x))
```
NumPy's [```gradient()```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html) function is used here to calculate the gradient from the run data (x, y) points.

### Putting it all together
With these functions defined, these metrics can be calculated using a variety of Python techniques, e.g. a list comprehension using Pandas' [```iterrows()```](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html) or using a lambda function with [```apply()```](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html):
```python
# calculate total power integral for trials...
integral_col_name = "total_power_normalized"
# ...for horsepower
hp_start, hp_end = summarize_trials_bounds(hp_trials['run_data'], 'horsepower')
hp_trials[integral_col_name] = hp_trials['run_data'].apply(lambda r: definite_discrete_integral(r, hp_start, hp_end))
# ...for torque
torque_start, torque_end = summarize_trials_bounds(torque_trials['run_data'], 'torque')
torque_trials[integral_col_name] = torque_trials['run_data'].apply(lambda r: definite_discrete_integral(r, torque_start, torque_end))

# calculate the 'max jump' i.e. biggest bump in horsepower for each hp trial
max_gradients = [max(discrete_gradient(r['run_data'])) for idx, r in hp_trials.iterrows()]
```

## Asking good questions
Simply put, a good question is one with a clearly defined quantitative measure. The calculus techniques discussed provide the base for analysis with empirical evidence - as long we are explicit about the assumptions and context that surround the calculation.

```filter_by_mods()``` allows us to filter trails by their modifications, so we compare apples to apples, and not apples to oranges. For example, a car with modified engine intake/exhaust or nitrous injection should not be compared to one without.

If you're an enthusiast in a hobby like cars, you may have noticed that people love to ask questions along the lines of _"What's the best part for the money?"_, or more simply _"What's the best part overall?"_. It's a compelling family of questions as old as the parts business, for any hobby.

Specifically with cars, the question is always **_"Who's the fastest?"_**, and it's never a bad question to start with!

### *Which run would be fastest in an 1/8th or 1/2 mile drag race?*
Drag races take place over 1/8th, 1/4, and 1/2 mile drag strips. 1/8 mile times are usually around 7 seconds - which means that the most important part of a drag race is right off the line. This means that we want the most "pull" off the line, i.e. the most torque.

The difference between 1/8th and 1/2 mile in this case is in what RPM range we care about - in the shorter 1/8th mile, you do not have the time to reach a higher RPM. So, for our project, we will define the ranges by statistical summary:

1/8th mile: [median start RPM, 4000]

1/2 mile: [median start RPM, 7000]

For each range, to find the best hypothetical drag car:
- **Calculate the total force for each torque run** by taking the integral on the specified bounds.
- Sort by the calculated total force, highest to lowest. **1st run will be the highest total force in the "off the line" RPM range**.


```python
# seperate copies for 1/8th, 1/2 mile
tq_trials_eighth = torque_trials.copy()
tq_trials_half = torque_trials.copy()
# get bounds
eighth_bounds = (summarize_trials_bounds(tq_trials_eighth['run_data'], 'torque')[0], 4000)
half_bounds = (summarize_trials_bounds(tq_trials_half['run_data'], 'torque')[0], 7000)
# calculate 'off the line' power for each
tq_trials_eighth['off_the_line'] = tq_trials_eighth['run_data'].apply(lambda r: definite_discrete_integral(r, eighth_bounds[0], eighth_bounds[1]))
tq_trials_half['off_the_line'] = tq_trials_half['run_data'].apply(lambda r: definite_discrete_integral(r, half_bounds[0], half_bounds[1]))
```

    torque start RPM summary: {'min': 1916, 'q1': 2400.0, 'q2': 2536.0, 'q3': 2882.0, 'max': 3300, 'outliers': [1012]}
    torque end RPM summary: {'min': 6399, 'q1': 6758.0, 'q2': 6961.0, 'q3': 7100.0, 'max': 7563, 'outliers': [7950, 6103]}
    torque start RPM summary: {'min': 1916, 'q1': 2400.0, 'q2': 2536.0, 'q3': 2882.0, 'max': 3300, 'outliers': [1012]}
    torque end RPM summary: {'min': 6399, 'q1': 6758.0, 'q2': 6961.0, 'q3': 7100.0, 'max': 7563, 'outliers': [7950, 6103]}
    

#### 1/2 mile
```python
num_top = 5
top = tq_trials_half.sort_values(by='off_the_line', ascending=False).head(num_top)
# plot the data
torque = [r['off_the_line'] for idx, r in top.iterrows()]
run = [r['name'] for idx, r in top.iterrows()]
y = np.arange(len(run))
plot.bar(y, torque, align = 'center', alpha = .7)
plot.xticks(y, run, rotation=35)
plot.ylabel('Total force (N)')
plot.xlabel('Run name')
plot.axis([-1,len(run), 2450000, 2625000])
plot.title('Off The Line Power ({} to {} RPM)'.format(half_bounds[0], half_bounds[1]))
plot.gcf().set_size_inches(12, 6)
plot.show()
```
<figure>
  <img src="{{site.url}}{{site.baseurl}}/static/images/dynoproject_halfmile.png" alt="Top 5 most powerful runs in the 1/2 mile RPM range."/>
</figure>

#### 1/8th mile
```python
num_top = 5
top = tq_trials_eighth.sort_values(by='off_the_line', ascending=False).head(num_top)
#plot the data
torque = [r['off_the_line'] for idx, r in top.iterrows()]
run = [r['name'] for idx, r in top.iterrows()]
y = np.arange(len(run))
plot.bar(y, torque, align = 'center', alpha = .7)
plot.xticks(y, run, rotation=35)
plot.ylabel('Total force (N)')
plot.xlabel('Run name')
plot.axis([-1,len(run), 560000, 620000])
plot.title('Off The Line Power ({} to {} RPM)'.format(eighth_bounds[0], eighth_bounds[1]))
plot.gcf().set_size_inches(12, 6)
plot.show()
```
<figure>
<img src="{{site.url}}{{site.baseurl}}/static/images/dynoproject_eighthmile.png" alt="Top 5 most powerful runs in the 1/8th mile RPM range."/>
</figure>

### *Which of the different types of fuel (pump gas, e85 ethanol fuel, 100 octane race fuel) provides the most power versus cost?*

Total power over each run has been calculated. For a fair comparison, filter out horsepower runs with:
- **stock engine** (stock heads, no methylene injection or nitrous)
- **same boost level** (either naturally aspirated/no turbo (0 boost) or most common, e.g. 20 psi - more boost means more power)
- **similar displacement** higher displacement means more power. Filter by similar ranges, e.g. [3.0, 3.1] or [3.5, 3.7].

On the topic of displacement, the range of 3.0 to 3.3 liters was chosen arbitrarily based on inspection of the data. The data set concerns a family of mechanically similar models - if the input data were of different cars, you might want a tighter range, say [3.0-3.05] liters, for a more stringent comparison.

...then group by the fuel type. For each group, find the ratio of total power over fuel cost per gallon. The highest ratio will be the most powerful fuel per dollar.
```python
gas_prices = {
    'pump':2.03,
    'race':13.11,
    'e85':1.83,
    # ed95 is ethanol diesel fuel, not commonly used - think e85, but for diesel cars. used in buses, etc.
    # because of its rarity, it is not commercially available - regular diesel price of $3/gal subsituted here.
    'e95m5':3.00
}
# boost level/displacement should not be too restrictive.
similar_boost_displacement_spec = {
    'boost':(10, 25),
    'heads':['stock', None],
    'injection':[None],
    'nitrous':[None],
    'displacement':(3.0, 3.3)
}
similar_boost_displacement = filter_by_mods(hp_trials, similar_boost_displacement_spec)
# calculate total power to fuel cost ratio for each trial
# if fuel is not specified, assume pump gas (which is a reasonable assumption!)
def power_cost_ratio(r):
    if 'fuel' in r['mods']:
        return r['total_power_normalized']/gas_prices[r['mods']['fuel']]
    else:
        return r['total_power_normalized']/gas_prices['pump']

power_cost_ratios = [power_cost_ratio(r) for idx, r in similar_boost_displacement.iterrows()]
similar_boost_displacement['power_fuelcost_ratio'] = power_cost_ratios
# split out fuel type as column so we can group by fuel type, again with default pump gas assumption
fuel_types = [r['mods']['fuel'] if 'fuel' in r['mods'] else 'pump' for idx, r in similar_boost_displacement.iterrows()]
similar_boost_displacement['fuel_type'] = fuel_types
# now group by fuel and show aggregate
fuel_grouped = {name:data for name, data in similar_boost_displacement.groupby('fuel_type')}
```
Power over cost ratios now calculated and grouped by fuel type, we average the runs' power/cost ratio over the runs using each type, and compare using the following bar chart.
```python
mean_ratios = similar_boost_displacement.groupby('fuel_type')['power_fuelcost_ratio'].mean().reset_index() 
# for labels
median_label_locs = np.arange(mean_ratios.shape[0])
# plot the data
w = 0.5
plot.bar(median_label_locs, mean_ratios['power_fuelcost_ratio'], width=w)

# title, labels, legend
plot.title('Fuel Type vs. Mean Total Power/Fuel Cost')
plot.ylabel('Total power (N)/Fuel Cost ($/gal)')
plot.xlabel('Fuel Type')
plot.xticks(median_label_locs, mean_ratios['fuel_type'])
# show
plot.gcf().set_size_inches(12, 6)
plot.show()
```
<figure>
<img src="{{site.url}}{{site.baseurl}}/static/images/dynoproject_fuel.png" alt="Fuel power vs. cost comparison."/>
</figure>

### *Which of the 10 most popular turbochargers has the least lag?*
Turbocharger lag is the "bump" in power delivery when boost kicks in. Looking at dyno charts, you can intuitively see the lag as those runs where are big "bumps" in the horsepower curve. To get the list of "bumps" in power delivery, we use the gradient from above!

The max element in the gradient is the maximum 'jump' - the max of the gradient for each comparable run will be the quantitative measure of turbo lag.

```python
# example of gradient for a single run
print_n_el = 5
run = hp_trials.iloc[0]['run_data']
print('Run data: {}...'.format(run[:print_n_el]))
print('Run gradient: {}...'.format(discrete_gradient(run)[:print_n_el]))
print('Max jump: {}'.format(max(discrete_gradient(run))))
```
```
Run data: [{'x': 2510, 'y': 53}, {'x': 2921, 'y': 107}, {'x': 2976, 'y': 109}, {'x': 3866, 'y': 214}, {'x': 4160, 'y': 257}]...
Run gradient: [0.13138686131386862, 0.0475788238577628, 0.04111365122601076, 0.13923603149124808, 0.16683480896161063]...
Max jump: 0.16683480896161063
```

So, to find the top 10 turbos by lag:
- **filter by same turbo class** - turbos are comparable by class - td03 turbo and td04 turbo is not a fair comparison.
- **filter by similar mods** (same heads, similar displacement)
- **find the max of the gradient for each run**
- **sort from least to highest max gradient value**, then take the top 10 **unique** turbos from the runs.


```python
# calculate the 'max jump' i.e. biggest bump in horsepower for each trial
max_gradients = [max(discrete_gradient(r['run_data'])) for idx, r in hp_trials.iterrows()]
hp_trials_copy = hp_trials.copy()
hp_trials_copy['max_jump'] = max_gradients
# filter to stock engine specs for fair turbo comparison
stockengine_spec = {
    'injection':[None],
    'nitrous':[None],
    'heads':['stock', None]
}
stock_engine = filter_by_mods(hp_trials_copy, stockengine_spec)
# filter by turbo classes t4, td04, td05 for fair comparison in each class
t4 = filter_by_mods(stock_engine, {'turbo_class':['t4']}) # no t4 turbos in our data set!
td04 = filter_by_mods(stock_engine, {'turbo_class':['td04']})
td05 = filter_by_mods(stock_engine, {'turbo_class':['td05']})
# report for each class: get turbo name as column so we can group by turbo...
```

#### TD04 top 10
```python
td04_turbos = [r['mods']['turbo_name'] for idx, r in td04.iterrows()]
td04['turbo'] = td04_turbos
td04_avg_maxlag = td04.groupby('turbo')['max_jump'].mean().reset_index().sort_values(by='max_jump')
# top 10
top = 10
print('Top 10 td04 turbos:')
display(td04_avg_maxlag.head(top))
# stats
print_df_summary(q_summary_dataframe(td04, 'max_jump'))
```
Top 10 td04 turbos:
<table border="1" class="dataframe half">
  <thead>
    <tr>
      <th>turbo</th>
      <th>max_jump</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9b</td>
      <td>0.100383</td>
    </tr>
    <tr>
      <td>gt357</td>
      <td>0.102240</td>
    </tr>
    <tr>
      <td>13g</td>
      <td>0.122219</td>
    </tr>
    <tr>
      <td>dr500</td>
      <td>0.141276</td>
    </tr>
    <tr>
      <td>phantoms</td>
      <td>0.148665</td>
    </tr>
    <tr>
      <td>17g</td>
      <td>0.166835</td>
    </tr>
    <tr>
      <td>dr650</td>
      <td>0.178770</td>
    </tr>
    <tr>
      <td>gt368</td>
      <td>0.217312</td>
    </tr>
    <tr>
      <td>15g</td>
      <td>0.217860</td>
    </tr>
    <tr>
      <td>dr650r</td>
      <td>0.286557</td>
    </tr>
  </tbody>
</table>

#### TD05 top 10
```python
td05_turbos = [r['mods']['turbo_name'] for idx, r in td05.iterrows()]
td05['turbo'] = td05_turbos
td05_avg_maxlag = td05.groupby('turbo')['max_jump'].mean().reset_index().sort_values(by='max_jump')
# top 10
top = 10
print('Top 10 td05 turbos:')
display(td05_avg_maxlag.head(top))
```
Top 10 td05 turbos:
<table border="1" class="dataframe half">
  <thead>
    <tr>
      <th>turbo</th>
      <th>max_jump</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GT28RS</td>
      <td>0.157058</td>
    </tr>
    <tr>
      <td>frank 50 custom</td>
      <td>0.231272</td>
    </tr>
    <tr>
      <td>14b</td>
      <td>0.238722</td>
    </tr>
    <tr>
      <td>GT3076R</td>
      <td>0.301879</td>
    </tr>
    <tr>
      <td>t2s</td>
      <td>0.307815</td>
    </tr>
    <tr>
      <td>5027s</td>
      <td>0.310863</td>
    </tr>
    <tr>
      <td>6765r</td>
      <td>0.312803</td>
    </tr>
    <tr>
      <td>GT3251</td>
      <td>0.330261</td>
    </tr>
    <tr>
      <td>evo3 16g</td>
      <td>0.330919</td>
    </tr>
    <tr>
      <td>dr1000</td>
      <td>0.344211</td>
    </tr>
  </tbody>
</table>

#### Comparison bar graph
```python
top = 10
td04_top = td04_avg_maxlag.head(top)
td05_top = td05_avg_maxlag.head(top)
# plot the data
w = 1
plot.bar([n * 2 for n in range(top)], td04_top['max_jump'], width=w, label='TD04', color='Red')
plot.bar([n * 2 + 1 for n in range(top)], td05_top['max_jump'], width=w, label='TD05', color='Blue')
# title, labels, legend
plot.title('TD04 top {} vs TD05 top {}: Boost Lag'.format(top, top))
plot.ylabel('Avg. top boost lag (d(hp)/d(rpm))')
plot.xlabel('Turbo')
plot.legend()
tups = list(zip(td04_top['turbo'], td05_top['turbo']))
labels = list(chain(*tups))
plot.xticks(ticks=list(range(top*2)),labels=labels, rotation=60)
# show
plot.gcf().set_size_inches(12, 6)
plot.show()
```
<figure>
<img src="{{site.url}}{{site.baseurl}}/static/images/dynoproject_turbos.png" alt="TD04 vs TD05 turbo comparison."/>
</figure>

## Conclusions
The drag race question, because it gives you a ranking of users based on their dyno charts, has value specific to the context in which it is applied. It could be used by a racing team aiming to perfect a drag racing car by comparing data for hypothetical prototypes.

The conclusion from the fuel comparison did not surprise me - often in discusson on forums for car modification, you will see e85 ethanol discussed as a cost-effective solution for high horsepower. The 85% ethanol/15% gasoline mixture is around 100 octane, similar to purpose-made race fuel - but much cheaper, especially in the Midwest where the corn used to produce the ethanol is plentiful.

The TD04 vs TD04 turbocharger comparison shows that TD05 turbos have higher boost lag - TD04 and TD05 describe different families of designs (as often happens in the parts world - orignially manufactured by Mitsubishi and copied/refined in the aftermarket) based on original model housing size. This would suggest that TD05 turbos have higher volume and take longer to spool.

As always, I have the data scientist complaints of lack of resolution and regularity in data.