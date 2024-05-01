# Course 5 TikTok project lab

# Course 5 - Regression Analysis: Simplify complex data relationships

# You are a data professional at TikTok. The data team is working towards building a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more eﬀiciently.
# The team is getting closer to completing the project, having completed an initial plan of action, initial Python coding work, EDA, and hypothesis testing.
# The TikTok team has reviewed the results of the hypothesis testing. TikTok’s Operations Lead, Maika Abadi, is interested in how different variables are associated with whether a user is verified. Earlier, the data team observed that if a user is verified, they are much more likely to post opinions. Now, the data team has decided to explore how to predict verified status to help them understand how video characteristics relate to verified users. Therefore, you have been asked to conduct a logistic regression using verified status as the outcome variable. The results may be used to inform the final model related to predicting whether a video is a claim vs an opinion.


# Course 5 End-of-course project: Regression modeling
# In this activity, you will build a logistic regression model in Python. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you’re measuring against. This opens the door for much more thorough and flexible analysis to be completed.
# The purpose of this project is to demostrate knowledge of EDA and regression models.
# The goal is to build a logistic regression model and evaluate the model.

# This activity has three parts:

# Part 1: EDA & Checking Model Assumptions * What are some purposes of EDA before con- structing a logistic regression model?
# Part 2: Model Building and Evaluation * What resources do you find yourself using as you complete this stage?
#Part 3: Interpreting Model Results
# What key insights emerged from your model(s)?
# What business recommendations do you propose based on the models built?



# Build a regression model
# PACE stages

# Throughout these project notebooks, you’ll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# PACE: Plan
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.


# Task 1. Imports and loading
# Import the data and packages that you’ve learned are needed for building regression models.

# Import packages for data manipulation

import pandas as pd
import numpy as np

# Import packages for data visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer from sklearn.utils import resample

# Import packages for data modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")



# PACE: Analyze
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.
# In this stage, consider the following question where applicable to complete your code response:
# What are some purposes of EDA before constructing a logistic regression model?

# Exploratory Data Analysis (EDA) before building a logistic regression model serves several purposes:
# Understanding Data Patterns: EDA helps in understanding the relationships between variables and identifying patterns in the data. This understanding is crucial for selecting appropriate variables for the model.
# Identifying Outliers and Missing Values: EDA helps in identifying outliers and missing values, which can affect the performance of the model if not handled properly.
# Feature Selection: EDA aids in selecting relevant features (independent variables) for the logistic regression model. It helps in identifying which variables are likely to have a significant impact on the outcome variable.
# Addressing Class Imbalance: In binary logistic regression where the outcome variable has two categories, EDA helps in understanding the distribution of the outcome variable and addressing any class imbalance issues if present.



# Task 2a. Explore data with EDA
# Analyze the data and check for and handle missing values and duplicates.
# Inspect the first five rows of the dataframe.

# Display first few rows
data.head()

# Get number of rows and columns
data.shape

# Get data types of columns
data.dtypes

# Get basic information
data.info()

# Generate basic descriptive stats
data.describe()

# Check for missing values
data.isna().sum()

# Drop rows with missing values
data = data.dropna(axis=0)

# Display first few rows after handling missing values
data.head()

# Check for duplicates
data.duplicated().sum()


# Create a boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_duration_sec', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_duration_sec'])
plt.show()


# Create a boxplot to visualize distribution of `video_view_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_view_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_view_count'])
plt.show()



# Create a boxplot to visualize distribution of `video_like_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_like_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_like_count'])
plt.show()



# Create a boxplot to visualize distribution of `video_comment_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_comment_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_comment_count'])
plt.show()



# Check for and handle outliers
percentile25 = data["video_like_count"].quantile(0.25)
percentile75 = data["video_like_count"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
data.loc[data["video_like_count"] > upper_limit, "video_like_count"] =␣ ↪upper_limit


# Check for and handle outliers
percentile25 = data["video_comment_count"].quantile(0.25)
percentile75 = data["video_comment_count"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
data.loc[data["video_comment_count"] > upper_limit, "video_comment_count"] =␣ ↪upper_limit


# Check class balance
data["verified_status"].value_counts(normalize=True)


# Use resampling to create class balance in the outcome variable, if needed
# Identify data points from majority and minority classes
data_majority = data[data["verified_status"] == "not verified"]
data_minority = data[data["verified_status"] == "verified"]


## With help: Upsample the minority class (which is "verified")

data_minority_upsampled = resample(data_minority,
replace=True, # to sample with␣
↪replacement ↪majority class ↪reproducible results
n_samples=len(data_majority), # to match␣ random_state=0) # to create␣


# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)

# Display new class counts
data_upsampled["verified_status"].value_counts()


# Get the average `video_transcription_text` length for claims and the average␣ ↪`video_transcription_text` length for opinions
data_upsampled[["verified_status", "video_transcription_text"]].
    groupby(by="verified_status")[["video_transcription_text"]].agg(func=lambda
    array: np.mean([len(text) for text in array]))

print(data_upsampled)


# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data_upsampled["text_length"] = (data_upsampled["video_transcription_text"].
                                 apply(func=lambda text: len(text)))


# Display first few rows of dataframe after adding new column
data_upsampled.head()



# Visualize the distribution of `video_transcription_text` length for videos posted by verified accounts and videos posted by unverified accounts
# Create two histograms in one plot
sns.histplot(data=data_upsampled, stat="count", multiple="stack",
             x="text_length", kde=False, palette="pastel",
                hue="verified_status", element="bars", legend=True)

plt.title("Seaborn Stacked Histogram")

plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for videos posted by
          verified accounts and videos posted by unverified accounts")

plt.show()



# Task 2b. Examine correlations

# Code a correlation matrix to help determine most correlated variables
data_upsampled.corr(numeric_only=True)

# Create a heatmap to visualize how correlated variables are

plt.figure(figsize=(8, 6))
sns.heatmap(
    data_upsampled[["video_duration_sec", "claim_status", "author_ban_status",
"video_view_count",
                "video_like_count", "video_share_count",
"video_download_count", "video_comment_count", "text_length"]]
    .corr(numeric_only=True),
    annot=True,
    cmap="crest")

plt.title("Heatmap of the dataset")

plt.show()





# PACE: Construct

# After analysis and deriving variables with close relationships, it is time to begin constructing the model.
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.



# Task 3a. Select variables
# Set your Y and X variables.
# Select the outcome variable.

# Select outcome variable
y = data_upsampled["verified_status"]

# Select features
X = data_upsampled[["video_duration_sec", "claim_status", "author_ban_status",
    "video_view_count", "video_share_count", "video_download_count",
    "video_comment_count"]]

# Display first few rows of features dataframe
X.head()




# Task 3b. Train-test split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    random_state=0)


# Get shape of each training and testing set
print('X_train.shape', 'X_test.shape', 'y_train.shape', 'y_test.shape')



# Task 3c. Encode variables

# Check data types
X_train.dtypes




# Get unique values in `claim_status`
X_train["claim_status"].unique()


# Get unique values in `author_ban_status`
X_train["author_ban_status"].unique()


# Select the training features that needs to be encoded
X_train_to_encode = X_train[["claim_status", "author_ban_status"]]
# Display first few rows
X_train_to_encode.head()


# Set up an encoder for one-hot encoding the categorical features
X_encoder = OneHotEncoder(drop='first', sparse_output=False)


#  Fit and transform the training features using the encoder
X_train_encoded = X_encoder.fit_transform(X_train_to_encode)


# Get feature names from encoder
X_encoder.get_feature_names_out()


# Display first few rows of encoded training features
X_train_encoded


# Place encoded training features (which is currently an array) into a dataframe
X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=X_encoder.
                    get_feature_names_out())


# Display first few rows
X_train_encoded_df.head()



# Display first few rows of `X_train` with `claim_status` and␣ ↪`author_ban_status` columns dropped (since these features are being transformed to numeric)
X_train.drop(columns=["claim_status", "author_ban_status"]).head()



# Concatenate `X_train` and `X_train_encoded_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `claim_status` and `author_ban_status`, so that the indices align with those in `X_train_encoded_df` and `count_df`
X_train_final = pd.concat([X_train.drop(columns=["claim_status",
                                                 "author_ban_status"]).reset_index(drop=True), X_train_encoded_df], axis=1)


# Display first few rows
X_train_final.head()


# Check data type of outcome variable
y_train.dtype


# Get unique values of outcome variable
y_train.unique()


# Set up an encoder for one-hot encoding the categorical outcome variable
y_encoder = OneHotEncoder(drop='first', sparse_output=False)


# Encode the training outcome variable

# - Adjusting the shape of `y_train` before passing into `.fit_transform()`, since it takes in 2D array
# - Using `.ravel()` to flatten the array returned by `.fit_transform()`, so that it can be used later to train the model

y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Display the encoded training outcome variable
y_train_final




# Construct a logistic regression model and fit it to the training set
log_clf = LogisticRegression(random_state=0, max_iter=800).fit(X_train_final,y_train_final)



## PACE: Execute

# Task 4a. Results and evaluation

# Select the testing features that needs to be encoded
X_test_to_encode = X_test[["claim_status", "author_ban_status"]]
# Display first few rows
X_test_to_encode.head()




# Transform the testing features using the encoder
X_test_encoded = X_encoder.transform(X_test_to_encode)

# Display first few rows of encoded testing features
print(X_test_encoded.head())



# Place encoded testing features (which is currently an array) into a dataframe
X_test_encoded_df = pd.DataFrame(data=X_test_encoded, columns=X_encoder.get_feature_names_out())


# Display first few rows
X_test_encoded_df.head()



# Display first few rows of `X_test` with `claim_status` and `author_ban_status` columns dropped (since these features are being transformed to numeric)
X_test.drop(columns=["claim_status", "author_ban_status"]).head()




# Concatenate `X_test` and `X_test_encoded_df` to form the final dataframe for training data (`X_test_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_test after dropping `claim_status`, and `author_ban_status`, so that the indices align with those in `X_test_encoded_df` and␣ ↪`test_count_df`
X_test_final = pd.concat([X_test.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True), X_test_encoded_df], axis=1)


# Display first few rows
X_test_final.head()



# Use the logistic regression model to get predictions on the encoded testing set
y_pred = log_clf.predict(X_test_final)



# Display the predictions on the encoded testing set
print(y_pred)



# Display the true labels of the testing set
print(y_test)



# Encode the testing outcome variable
# Notes:
# Adjusting the shape of `y_test` before passing into `.transform()`, since it takes in 2D array
# Using `.ravel()` to flatten the array returned by `.transform()`, so that it can be used later to compare with predictions

y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()

# Display the encoded testing outcome variable
print(y_test_final)


# Get shape of each training and testing set
X_train_final.shape, y_train_final.shape, X_test_final.shape, y_test_final.shape


# Task 4b. Visualize model results

# Compute values for confusion matrix
log_cm = confusion_matrix(y_test_final, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()


# Create classification report for logistic regression model
target_labels = ["verified", "not verified"]
print(classification_report(y_test_final, y_pred, target_names=target_labels))


# Task 4c. Interpret model coeﬀicients

# Get the feature names from the model and the model coefficients (which represent log-odds ratios)

# Place into a DataFrame for readability
pd.DataFrame(data={"Feature Name":log_clf.feature_names_in_, "Model Coefficient":log_clf.coef_[0]})



# Task 4d. Conclusion
# 1. What are the key takeaways from this project?
# 2. What results can be presented from this project?

# Some variables in the dataset are strongly related, which might cause problems when making a logistic regression model.
# We left out the number of likes on videos when making the model.
# According to the model, every extra second of the video makes it a bit more likely for the user to be verified.
# The model we made predicts okay: it's not great, but it's not bad either.
# It's right about 61% of the time, which could be better, but it's really good at finding the right cases about 84% of the time.
# We made a model to guess if someone is verified based on video stuff. It works okay. Longer videos are more likely to belong to verified users, but other video details don't seem to matter much.

