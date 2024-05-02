# Machine Learning in Strawberry Puree determination
The use of spectrography techniques in food science and chemometrics is often employed for food types classification
and detection, a task that has important applications in food safety, authenticity assessment, and quality assurance.
The file contains a total collection of 983 mid-infrared spectra collected from different
fruit purees. Each spectrum is assigned to one of two classes: strawberry, purees prepared from fresh whole strawberries by the food scientists, and adulterated, diverse collection of other purees (raspberry, apple, blackcurrant,
blackberry, plum, cherry, apricot, grape juice, and mixtures of these) or strawberry adulterated with other fruits and
sugar solutions. The researches wish to build
a supervised learning system capable of discriminating pure strawberry purees from adulterated or fake purees. 
The following classifiers are considered and compared in order to predict if a sample is pure strawberry or not using
the input spectrum features:
• 𝐶1 – Standard logistic regression classifier + PCA dimension reduction with 𝑄 coordinate vectors.
• 𝐶2 – Regularized logistic regression model with 𝐿1 penalty function
