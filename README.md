# PROJECT
## INFO
This repo contains our implementations of the CHE 4230 Advanced Process Control Systems – Spring 2023 Semester Project. The project has a plant setup as shown below:

![Screenshot](./reports/flowdiagram.png)

##  GROUP MEMBERS 
>**1. Ana B Terreros**\
>**2. Amanda M Ross**\
>**3. Teslim O. Olayiwola**
## CONTENT
>**data** contains the data used in this study.\
>**notebooks** contains the all the jupyternook.\
>**models** contains the trained and serialized models, model predictions and summaries\
>**src** contains the python modules. \
>**docs** contains the other non-related PDF docs. \
>**reports** contains the analysis (PDF) and figures.
## HOW TO USE
```python
# install package
pip install TEPaat
# use package
from TEPaat import TEPaat
test = TEPaat()
# load data
data = test.load_data(path_to_featuredata, no_toplabel = False) # no_toplabel = False means there is a column name
labels = test.load_data(path_to_label_data, no_toplabel = False) # please ensure that excelsheet has a column name 'label'
# predict class 
## Note that the code assumed that the labels has: {2, 3, 6, 8, 13}.
## The code has converted this labels based on our analysis that {2, 3, 6, 8, 13} is equal to {1, 3, 4, 0, 2}
predictions = test.predict(data)
accuracy = test.eval_accuracy(predictions, labels)
print(f'The performance of the model is: {accuracy}')
```
