# tabular-playground-09-21
This is my first repo. It contains my code for the Kaggle playground competition series for the month of September. 

In this read me doc, I will explain my long-form process with a summary at the end.  

**Entry**
Choosing this competition was a simple matter. It is a challenge for beginning data scientists, I am a beginner in the field, and, wanting something I could compare myself to others in, it just made sense. 

When you look at a Kaggle competition prior to joining there isn't much usable information. Looking back I don't remember what they offer except that this would be playground set. In this case, that meant that it was made up. The numbers aren't real. I could also know that it is based on insurance claim data, there were roughly 120 columns, and many, many rows. 

Something I was warry of in entering this competition was the size of the data set. The larger the data set the longer the processing time. This has turned out to be one of my bigger challenges. Later you'll see how I dealt with that problem.

Next I officially signed up for the competition. I agreed to the terms and, having done that, I could download the data.

**Beginning**

I download my datasets to a particular folder called, big surprise here, "data sets." This keeps things simple. I know where to look for all my data sets when I use a local notebook, or even an online notebook, and the path to the files is reliably similar. 

Once downloaded I open my local Jupyter notebook and take a look at the csv file.

This first requires that I import my pandas library. Since I'll be importing that library, I'll import the others here as well. 

(The libraries consist of numpy, pandas, sklearn, matplotlib, pdpbox, and shap.) 

df = pd.read_csv('train.csv')
df.head()

**Claim**

I can see clearly that there are 120 columns, most of which are titled f1-f118. The final row is set apart with the title 'claim.' Evident from the format, 'claim' is my target value; it is what everyone in the competition will be trying to predict. (It also says in the description that this is the case.) Furthermore, the target column is set out as a binary classification, that is, if the the claim was approved it is labeled a 1 and if the claim wasn't approved it is labeled a 0. If we are to predict whether the claim is approved or not, this is a classification problem. If, however, we are predicting the probability of the claim we could have a regression problem. 

**Wrangle**

The final column helped us with the target. The first column will be the index for us. It is labeled 'id.' What is nice about 'id' is that all the values are unique and identifying for the rows. 

While checking the values for 'id' I go ahead and look check for missing values.

My hypothesis for missing values is that they would negatively affect the claim. In order to test this I'll create a feature called 'null_values.' In order to make this feature I transform the columns into rows and take the sum of missing values in the new columns. 

The next thing to stand out to me is that the values of the data set vary greatly from e15 numbers to .0000001. Because these values are relatively well distributed throughout the columns, it makes sense to me to scale the data. In order to scale the data I will fist take out the claim column, I don't want that scaled. Next I will use a min-max normalization to scale the data. The formula runs: df - df.min / df.max - df.min. Afterwords we concatenate the claim column back onto the data frame.

Another thing I'll do is, instead of deleting the rows with the missing values, impute them with the mean of the columns. This allows us to run the data set through a model without nan errors. Since we have the number of nans per row in a column we can still take into account the affect of the missing rows on the rows themselves. 

**Target Distribution**

In order to find the target distribution we take the target column and find the normalized distribution of 1s and 0s. 50.2% of the claims were not approved and 49.8% of the claims were approved. 

**Split Data**

At this point I will split the data into our matrix, X, and our target variable, y. This will essentially allow us in the next step to separate our data into training and validation sets.

**Train test/val split**

In the previous step we separated the target column from the rest of the data set. In this step we are going to split the X and y variables into our training and our validation sets. This will allow us to run a training data set to train our data and then validate our results, or see how accurate and generalizable our training data was. 

**Linear Regression Model** 

Setting up these models is relatively simple. I first instantiate the model and then fit it to my training set. 

**Logistic Regression Model**

Like the linear model, I instantiate the model and then fit to the training set. 

**Decision Tree Model**

Here is where I start running into problems with the size of my data set. The next few models, tree models and boost models, all take too long to process on google colab, which is where I prefer to do my work. It takes so long that I can't get results. 

**Audible**

I still need something to train my data, however, and perhaps later I can test the models in a local notebook where I have faster processing times. For now my solution to the problem of a huge data set is to take a random sample that is manageable in size. 10000 is the size I choose to deal with. 

**Decision Tree Model**

**Random Forest Classifier Model**

**Random Forest Regressor Model**

After taking the small sample of the data set, running the tree based models went quickly.The random forest only taking 2 min. 

**Boosted model**

Using XGBoost took 36 seconds to run.

**Evaluation Metrics**

Now that I have instantiated and fit all of my models I can test the metrics. The evaluation metrics I am using are: mean absolute error, r^2 score, and accuracy. In regression models the code .score returns the r^2 score, whereas with classification models we are returned the accuracy. So we will see some variation in the metrics, this is due to the different scores that are being used. 

The evaluation metrics that look most promising are the XGBoost and the Random Forest Classifier.

**Hyper-Parameter Tuning**

In order to tune the hyper-parameters I chose to you the Randomized Search CV as opposed to grid search for processing's sake. 

First step was to choose the models I wanted to tune, random forest classifier and XGboost. 

Then I wanted to set up a grid of the parameters I wanted to tune and the settings I wanted to try out. 

In my parameter grid for random forest I chose to try out: Max Depth from 3-7 by 1s, n_estimators from 100-500 by 100s. For the boosted model I simply used max_depth from 3-7 by 1s. 

During this process I learned that in order to do the randomizedsearchcv, the model must be in a pipeline. Otherwise I got an error.

After minimal hyper-parameter tuning the random forest classifier still came out on top.

**Testing testing**

After this, comes testing. Running the models I have created on the training and validation sets on the test set. creating a csv of the probability and submitting that to Kaggle. 
