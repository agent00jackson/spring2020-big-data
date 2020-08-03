## Introduction
With the advent of technology and electronic transactions, the need for security and protection against fraud becomes imperative, especially for credit companies which are reliant on consumer purchasing habits and must operate while maintaining their customers’ trust by protecting their money as well as ensuring the least amount of fraud disputes.  Given the number of electronic payment transactions daily in the billions, banks must make use of machine learning techniques to ensure a high level of accuracy in detecting fraud and notifying customers as soon as possible to take action.  This is a binary classification problem: transactions are either fraudulent (1) or legitimate (0).  In this project, we explored and compared two of the various techniques that aid in fraud detection: Logistic Regression and Random Forest Classifier. 

## Dataset
We used a data set with over 280 thousand transactions from European cardholders, comprised of 29 features, including time.  Only 0.172% of the transactions were fraudulent, which made the data set highly skewed. The transactions were collected in September of 2013 through a collaboration between Worldline, the Machine Learning Group, and The Free University of Brussels (Université libre de Bruxelles). 

Kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud  

## Methodology
We implemented both Logistic Regression and Random Decision Forest classification and compared the results. We chose Logistic Regression due to our experience with it in the class and due to its applicability to binary-classification, while we chose to experiment with random forests because they seemed popular to use in our background research.

We did not have to engineer the dataset, all the features except "Time" and the "Amount" had been transformed using Principal Component Analysis (PCA) prior to applying our training. 

By applying the Random Decision Forest model, given the seemingly unrelated and anonymous nature of the 28 features provided, it seemed appropriate that the algorithm builds Decision Tree subsets at random.  We trusted the default parameters, using the "gini" impurity parameter over entropy and leaving the max decision trees set for 20. 

### Undersampling and Oversampling
After writing and running the code to train our logistic regression model, we noticed that the performance statistics, other than accuracy, were at their worst-case values. The authors of the dataset mentioned this might happen due to the unbalanced data. Common practices we researched were under and over-sampling of the data. 

#### Undersampling
We implemented a random undersampling strategy. We used the built-in spark API to randomly sample data points from the majority class, which in our case, was legitimate transactions. We immediately noticed an improvement in the logistic regression model's performance.

![Undersampling Diagram](https://i.imgur.com/0ClFIJ6.png "Undersampling Diagram")

#### Oversampling with SMOTE
Oversampling is creating new data points of the minority class. The simplest way to perform oversampling is to duplicate data points, however this has the drawback that our models could simply "memorize" certain data points, leading to severe overfitting. One of the popular strategies to deal with this is the Synthetic Minority Oversampling Technique (SMOTE).
SMOTE uses a K-Nearest-Neighbors strategy, where it takes K nearest neighbors and generates points in between them. Instead of coding our own implementation of SMOTE, which isn't a part of the default Spark API, we used a community created implementation that utilizes the Spark SQL API:
https://github.com/alivcor/SMORK

![SMOTE Illustration](https://i.imgur.com/zDHuJn9.png "SMOTE Illustration")

## Results
### Logistic Regression
![Logistic Regression Results](https://i.imgur.com/eRoz6Rt.png "Logistic Regression Results")

### Random Forest Classifier
![Random Forest Classifier Results](https://i.imgur.com/YtCow8O.png "Random Forest Classifier Results")

## Comparisons and Conlusions
In terms of accuracy, all techniques performed well (each above 98% accuracy), however accuracy isn't the most important metric in this application.  With regards to Precision and Recall which compute the F1 score, Logistic Regression and Undersampling performed much better than without resampling while Random Decision Forests classifier performed even higher without resampling as well as with SMOTE and Undersampling. Our AUC ROC probability for a positive class prediction increased to 97% in the logistic regression model. In conclusion, using the Random Forest Classifier with SMOTE and Undersampling gave the best results. While other models had higher overall accuracy, they all sacrificed recall, which is our most important measure of performance in this application.

## Contributions
Houida Aldihri — Contributed to editing the group documents
Brad Barnes — Contributed to editing the group documents and discussing the background and dataset slides during the presentation
Colleen Hynes — Contributed to Random Decision Tree Classifier program code, Decision Tree slides of presentation, editing group documents
Aidan Murphy — Contributed to editing the group documents and discussing the implementation of logistic regression vs random forest classifier during the presentation
Jackson Randolph — Contributed to editing the group documents, outline, topic source selections, implementing most of the Scala code, and discussing the undersampling and the oversampling techniques as well as the conclusion slides of the presentation

## References
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

“Credit Card Fraud Detection - An Insight Into Machine Learning and Data Science.” 3Pillar Global, 30 July 2019, www.3pillarglobal.com/insights/credit-card-fraud-detection.

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Chawla, N. V. et al. “SMOTE: Synthetic Minority Over-Sampling Technique.” Journal of Artificial Intelligence Research 16 (2002): 321–357. Crossref. Web.

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

“How to Build a Random Forest Classifier Using Data Frames in Spark.” Learning Tree Blog, 12 Nov. 2015, blog.learningtree.com/how-to-build-a-random-forest-classifier-using-data-frames-in-spark/.

Patel, Savan. “Chapter 5: Random Forest Classifier.” Medium, Machine Learning 101, 18 May 2017, medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1

Spark 3.0.0-preview2 ScalaDoc - Org.apache.spark.ml.evaluation.BinaryClassificationEvaluator, spark.apache.org/docs/3.0.0-preview2/api/scala/org/apache/spark/ml/evaluation/BinaryClassificationEvaluator.html.
