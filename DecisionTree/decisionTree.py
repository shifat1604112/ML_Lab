import pandas as pd
df = pd.read_csv("depression.csv")
df.drop(['Whichyear','Gender','Yourlocation','happywithlivingplace','donerecreationalactivitytoday','Age','Relationshipstatus', 'Understandingwithfamily','feelingpressureinyourstudy','supportsyouyouracademiclife','usedanysocialmedia','satisfiedwithmeal','feelingSick/healt issues'], axis=1, inplace=True)
df

training_data = df.values[:,0:]
training_data
header = ["feelingrightnow","ExpressFeeling","Areyouhappyinancialy","succeededInEducationalinstitution","satisfiedwithacademicresult","haveinferioritycomplex","sleepAtNight"]

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

unique_vals(training_data, 0)

def class_counts(rows):
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class_counts(training_data)