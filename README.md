# This is an Implementation of ID3

ID3 is a decision tree algorithm which uses information gain (derived from entropy) to branch out.
the main idea is to maximize entropy at each node (divide in equaly sized groups).

https://en.wikipedia.org/wiki/ID3_algorithm

--- 

# How to use

```python
df = pd.DataFrame(
    {
        "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
        "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild",],
        "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","High","High"],
        "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
        "PlayGolf": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"],
    }
)

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
target_attribute = "PlayGolf"

tree = DecisionTree()
tree.fit(df, attributes, target_attribute)

x = df.iloc[0].to_dict()
tree.predict(x)
```

## Notes: 
* every column should be prepocessed to contain only categorical features
* the fit method will destroy previous training
* the node implementation uses the values to traverse the tree, new values different from the train set will raise an exception 
* values used for inference should be dicts