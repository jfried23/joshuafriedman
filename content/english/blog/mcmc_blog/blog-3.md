---
title: "How Sure Are You? -- Metropolis–Hastings Algorithm"
date: 2021-02-16T12:14:34+06:00
image: "images/blog/003_mcmc/launch.jpg"
tags: [Baysian, Metropolis–Hastings algorithm]
description: "This is meta description."
draft: false
---

Baysian statistics provide a really elegant framework for understanding your statistical models. The problem is that darned normalization $P(B)$ constant from Bayes famous equation 
$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{ P(B) }$$ 
in the general case where the dimensionality of A may be very large, we don't know how to evaluate $ P(B) = \int P(B|A) dA $ given the joint probability.

#### Lets Make Things A Bit More Concrete Shall We?
The Space Shuttle's two solid rocket boosters (SRB) contained rubber O-Rings that were susceptible to damage when the ambient air temperature dropped in Cape Canaveral Florida. [We will be using a dataset documenting post launch damage inspection of 23 launches of the space shuttle as a function of temperature](https://www.openintro.org/data/csv/orings.csv) to motivate this work. 

Given this dataset we will build a simple logistic regression model using our temperature feature to understand the probability of O-Ring Failure:

$$ P( Failure | m, b ) = \frac{1}{1 + e^{-(m \cdot temp + b)}}$$

where 'm' is our slope and 'b' the y-intercept of the graph. We can quickly fit this via sckitlearn:

{{< highlight py3 "linenos=table,linenostart=1, hl_lines=0" >}}
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Read the data
data = pd.read_csv('./data_csv')
temp = data.temp.values.reshape(-1,1)
fail = (data.damage>0).astype('int').values


model = LogisticRegression()
model.fit( temp, fail )

print("m: {:.4f}\nb: {:.4f}".format(model.coef_.item(), model.intercept_.item()))
{{< / highlight >}}
{{< highlight py3>}}
    m: -0.2295
    b: 14.8619
{{< / highlight >}}

And after evaluating our fit:

![figure](/images/blog/003_mcmc/niave.png)

#### But how confident should we be about $m$ and $b$?
Great question! Lets be Baysian about our model parameters! We have now defined a plausible function for evaluating $P( Failure | m,b )$ above, but we really want to do now is characterize $P( m,b | Failure )$.

The Metropolis–Hastings algorithm provides us an opportunity to do just that using knowledge $P( Failure| m,b)$ alone! How cool is that? Here is how the Metropolis–Hastings algorithm works in pseudo code:

> Initialize $m$ and $b$ to some starting value. Then for each step 't' in our simulation:
>   1. Generate a candidate values for $m'$ and $b'$ according to some transition function $ g(m,b) \rightarrow m',b'$
>   2. Calculate the acceptance probability $ \alpha = min( 1, \frac{P(Failure | m',b')}{P( Failure | m,b)}) $
>   3. Draw a random value $v \sim Uniform(0,1)$
>       1. If $ v \leq \alpha $, accept the draw and set $m=m'$ and $b=b'$
>       2. Otherwise continue.

By following this procedure it can be demonstrated that the chosen values of $m$ and $b$ are drawn from our unknown distribution $P( m,b | Failure )$! Lets code this up shall we?

First lets create function for generating candidate values $ g(m,b) \rightarrow m',b'$:
{{< highlight py3 "linenos=table,linenostart=1, hl_lines=0">}}
import numpy as np

def markov_step( inital_m, inital_b ):
	"""
	Utility function for proposing new values of m, b
	given their current value using Normal distribution.
	
	INPUTS:
		inital_m -- (float) starting value of slope 'm'
		inital_b -- (float) starting value of y-intercept 'b'
	RETURNS:
		tuple of proposals
	"""
	proposed_m = np.random.normal(inital_m,0.001)
	proposed_b = np.random.normal(inital_b,0.001)

	return proposed_m, proposed_b

{{< / highlight >}}

![figure](/images/blog/003_mcmc/randomsteps.png)

Great now we need a function to calculate the likelihood of our Failure data set given the proposed values of $m$ and $b$. The logistic function model we defied gives us the P(Failure | m,b) and Bernoulli distribution will define the likelihood of our failure data given the model predictions (we will use log-likelihood here to control machine precisions).

{{< highlight py3 "linenos=table,linenostart=18, hl_lines=0">}}

def model( temp, m, b ):
	"""
	Evaluates P(Failure| m,b) as a function of temp

	INPUTS:
		temp -- (np.array<float>) the temperatures to evaluate our model at
		m    -- (float) the slope
		b    -- (float) the y-intercept

	RETURNS:
		numpy array recording the probability of failure at each temperature
		given model parameters.
	"""

	coeff = temp*m + b
	return 1/(1+np.exp(-coeff))

def binomial_log_likleyhood( failure, predictions ):
	"""
	Calculates the Bernoulli log-likelihood for a series of iid datapoints.

	INPUTS:
		failure -- (np.array<boolean>)  the observational training data
		predictions -- (np.array<float>) 

	RETURNS:
		float value representing the sum of the data log-likelihoods given the
		model predictions.
	"""
	predictions = predictions.flatten()
	failure = failure.flatten()
	liklehood = failure*predictions + (1-failure)*(1-predictions)
	log_liklehood = np.log(liklehood.clip(min=1.e-12))
	return log_liklehood.sum()

{{< / highlight >}}

With these componets in place, all that is left is a simple function implementing the Metropolis–Hastings algorithm controll logic.

{{< highlight py3 "linenos=table,linenostart=18, hl_lines=0">}}

def metropolis_hastings(num_steps, model, temp, failure_data):
	
	#initalize values for m and b
	proposed_m, proposed_b = -0.2,14.8
	sample_trace = np.zeros((num_steps,2), dtype=np.float)

	step=-1
	while step < num_steps:
		proposed_m, proposed_b = markov_step(proposed_m,proposed_b)
		model_predictions = model( temp, proposed_m, proposed_b )
		proposal_likelihood = binomial_log_likleyhood(failure_data, model_predictions)

		if step == -1:
			current_likelihood = proposal_likelihood
			step+=1
			continue

		acceptance_cutoff = np.random.rand()
		alpha = np.exp(proposal_likelihood - current_likelihood)
		alpha = np.clip(alpha,0,1)

		if alpha >= acceptance_cutoff:
			sample_trace[step] = [proposed_m, proposed_b]
			current_likelihood = proposal_likelihood
			proposed_m, proposed_b = markov_step(proposed_m, proposed_b)
			step+=1
	return sample_trace

{{< / highlight >}}

![figure](/images/blog/003_mcmc/samples.png)

Now takeing these sample traces we can plot the range of values of $m$ and $b$ that are consistent with our model with a 95% confidence interval!

![figure](/images/blog/003_mcmc/range.png)
***Did our little experiment work?*** 

Kinda sorta just barely...not really no to be honest. Our 'markov_step' function for generating proposal samples is super wasteful because it has no information about the energy landscape of our function and its step size needs to be hand tuned to accommodate unscaled features. Because of proposal function is so sensitive to step size our samples are not particularly well mixed and our drawn samples are not fully exploring the energy landscape, thus our draws only poorly approximate the true posterior of $P(m,b|data)$. 

What we really need is better way to propose new values of 'm' and 'b'; enter [Hamiltonian Monte Carlo methods!](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) Lets implement from scratch in a future post (this one is already running long in the tooth) but for now lets just solve our problem with a legit tool kit like [numpyro](https://github.com/pyro-ppl/numpyro) which uses jax for it's automatic differentiation support.

{{< highlight py3 "linenos=table,linenostart=0, hl_lines=0">}}

import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist

def srb_model(temp,fail):
    b = numpyro.sample('b', dist.Uniform(0,100))
    m = numpyro.sample('m', dist.Normal(0,10))
    
    with numpyro.plate('d',temp.size):
        coeff = b+m*temp
        prob  = 1/(1+jnp.exp(-coeff))
        
        numpyro.sample('fail',dist.Binomial(probs=prob), obs=fail)
        
nuts_kernel = numpyro.infer.NUTS(srb_model)
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
rng_key = jax.random.PRNGKey(0)

temp = jnp.array(temp, dtype=jnp.float32).reshape(-1,1)
fail = jnp.array(fail,dtype=jnp.float32).reshape(-1,1)

mcmc.run(rng_key, temp, fail)
mcmc.print_summary()
{{< / highlight >}}

{{< highlight py3 >}}
                mean       std    median      5.0%     95.0%     n_eff     r_hat
         b     15.21      1.65     15.10     12.76     18.10    186.22      1.00
         m     -0.23      0.02     -0.23     -0.28     -0.20    184.28      1.00

Number of divergences: 0
{{< / highlight >}}

Much nicer / easier!
