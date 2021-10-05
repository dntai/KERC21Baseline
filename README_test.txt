This data release includes the 'Private Test' dataset. This will be used for private leaderboard on Kaggle, 
while previously provided validation data will be used for public leaderboard. Therefore,the submissions to Kaggle 
should include results for both validation data(public test) and test data(private test) in a single CSV file.
The samples in submission file can be in any order. (ie. both public test and private test can be mixed together)

--------------------------------------------
File Information
-------------------------------------------

KERC21Dataset_test.zip
- Test Dataset 
- personality CSV file

----------------------
Supplimentary Code 
---------------------
During test data release: following changes were made to the baseline code provided before through Kaggle.
- dataset.py
	-  Reading sample ids from the train labels(to avoid iterating over folders) when loading train data.
	-  And for val(public test) and test(private test) dataset, read sample ids from directory names.
- generate_submission.py
	- Merging two different test sets for Kaggle Submissions.
 
