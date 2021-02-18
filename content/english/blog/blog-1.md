---
title: "CVXPY For Classifcation"
date: 2021-02-16T12:14:34+06:00
image: "images/blog/001_cvxpy/shrodinger.png"
tags: ["classfication","discrete optimization","CVXPY"]
description: "This is meta description."
draft: false
---

Classification models typically compute the likelihood a data point belongs to a set of predefined categoricals, but to be useful we typically need to arrive at an absolute definitiative assignment at the end, not a vector of probabilities.
Applying a classification models thus requires an additional step apart from building/training the model; the process of binarizing (0/1) the model output probabilities into definitive class assignments. For the simplest of classification tasks taking the argmax of each prediction vector might be good enough, but I have seldom encountered these types of problems as a practicing data scientist!
Real world classification problems might require outputted assignments to:

1. Make sense in aggregate. – e.g. an income stratification model that never assigned people to the 10-percentile bucket might raise some eyebrows even if it does maximize individual likelihoods.[^1]

2. Deal with possibility of non mutually exclusive class labels (muti-label classification). – e.g. movies often mix multiple genre labels, ‘comedy’ and ‘horror’.

Optimizing these types of problems means discrete optimization! I plan on covering the branch-and-bound algorithm for discrete optimization in subsequent posts, but here I will instead focus on practical usage via the python packages [CVXPY](https://www.cvxpy.org/) and [cvxopt](https://cvxopt.org/). 

Confusingly the CVX in CVXPY stands for ***convex*** optimization, and discrete optimization is a decidedly ***non***-convex objective. What it laks in appropiate naming however CVXPY more than makes up for with its elegent API for expressing constrained optimization objectives. The package seamlessly wraps multiple branch-and-bound solver backends including [ECOS-BB](https://github.com/embotech/ecos) and [GLPK-MI](https://www.gnu.org/software/glpk/), freeing us from having to use the unique ~~clunky~~ quirky interface of each package without ever having to leave python! Lets see it in action using a toy problem!
 
######  CVXPY for argmax: a toy problem
Lets say we have an built a model to classify images as containing Dogs, Cats, Birds or Fish. We then apply our model to 5 new images and get class output probabilities represented as veriable "A" in the code excerpt below. Now we could use argmax here for digitization, but whats the fun in that? Lets try to solve it with CVXPY just so we can test out the API as a first pass. 

First we will define a masking variable "x" in CVXPY with the same shape as our probabilities "A" but with the keyword arg "*integer*" set to True; informing CVXPY that this "x" will be constrained to integer values.    

{{< highlight py3 "linenos=table,linenostart=1, hl_lines=13" >}}
import cvxpy
import numpy as np

A = np.array(
      # classes:    Dog,    Cat,   Bird,   Fish                
               [[0.3838, 0.1096, 0.2164, 0.2901],
                [0.0232, 0.1819, 0.5227, 0.2722],
                [0.0333, 0.2398, 0.1939, 0.5330],
                [0.2575, 0.3151, 0.0349, 0.3925],
                [0.1833, 0.1070, 0.4172, 0.2925]]
                )

x = cvxpy.Variable(shape=probabilities.shape, boolean=True)
{{< / highlight >}}

<p></p>
Our boolean variable "x" will contain our assignment for each image, so we wish to find the value of "x" that will be non-zero in the location of the largest categorical probability for each image (row). This objective can be expressed as 

$$ \underset{ x \in \mathbb{Z}^{+} }{\operatorname{argmax}} \sum A * x$$

where the $*$ symbol denotes elementwise multiplication and the summation is over all values. This can be susintly expressed in CVXPY using the built in atomic function "sum" and "multiply" below. A full list of [supported functions in CVXPY can be found here](https://www.cvxpy.org/tutorial/functions/index.html). With our likleyhood expression in hand we can then specify our maximization objective in line 15 below:

<p></p>

{{< highlight py3 "linenos=table,linenostart=14" >}}
likelihood = cvxpy.sum( cvxpy.multiply( A, x ) )
objective = cvxpy.Maximize(likelihood)
{{< / highlight >}}

Now if we were to try and solve the objective as defined above we would get a matrix "x" containing all ones. Fortunately we have some meta-data about our problem, we know that each image (row) has only a single class label $\in$ (Dog, Cat, Bird or Fish). We provide this information to our optimizer by specifing a single logical constraint upon "x" specified on line 16 below; the sum of the rows in x must sum to exactly one. Thats all folks!    

We construct our "Problem" object by passing it an objective (defined above) and a list of constraints and use its solve method.

{{< highlight py3 "linenos=table,linenostart=16" >}}
constraint = [ cvxpy.sum(assignments,axis=1) == 1.0 ]

defined_problem = cvxpy.Problem(objective, constraints = constraint)

defined_problem.solve(solver=cvxpy.GLPK_MI)

{{< / highlight >}}

<p>You can access the optimized value of "x" using the "value" method like so below, and voilà!</p>

{{< highlight py3  >}}
np.round(x.value)

array([[1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.]])

{{< / highlight >}}
<p>Sure we may have just preformed an unnecessarily complex argmax; but have done so in a prinicpaled way that will beatifully generalize to other more complex problems.</p>

######  Using Information About Known Class Frequencies
As we can see from our solution above no images were assigned the label cat (column #2). But what if we had some additional meta-data about problem; lets say we are told every class label is represented at-least once in our 5 member image set. Our goal is no longer to find the most likley class for each image, but to find the most likley global assignment amongst all the images subject to our new constraint.

CVXPY makes this easy, we just need to add this new constraint to our problem definition above:

{{< highlight py3 "linenos=table,linenostart=16, hl_lines=17" >}}
constraint = [ cvxpy.sum(assignments,axis=1) == 1.0,
               cvxpy.sum(assignments,axis=0) >= 1.0
             ]

defined_problem = cvxpy.Problem(objective, constraints = constraint)

defined_problem.solve(solver=cvxpy.GLPK_MI)

{{< / highlight >}}

Our new logical constraint on line 17 specifes that the summing our assignments over the image dimension must produce at-least one assignment for each the four categoricals. Running this again generates the new optimal solution:

{{< highlight py3 >}}
#     Dog, Cat, Bird, Fish  
array([[1.,  0.,  0.,  0.],
       [0.,  0.,  1.,  0.],
       [0.,  0.,  0.,  1.],
       [0.,  1.,  0.,  0.],
       [0.,  0.,  1.,  0.]])

{{< / highlight >}}

where the class assignment for the 4th image has been switched from fish to cat. 

If we were told our image set contained only dog and cat images, no sweat:


{{< highlight py3 "linenos=table,linenostart=16, hl_lines=18-19" >}}
N = A.shape[0]
constraint = [ cvxpy.sum(assignments,axis=1) == 1.0,
               cvxpy.sum(assignments,axis=0) >= [1,1,0,0],
               cvxpy.sum(assignments,axis=0) <= [N,N,0,0]
             ]

{{< / highlight >}}
<p>Which gives:</p>
{{< highlight py3  >}}

array([
  #     Dog, Cat, Bird, Fish 
       [ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.]])

{{< / highlight >}}




[^1]: Exploting data class imblances makes me cringe too.... I will save this rant for later but in business operating on available data ~~sometimes~~ oftentimes needs to take precedence over acquiring data that is actually predicate of anything. Thats an unpleasent reailty for  a data**scientists** but what can I say? *C’est la vie*.
