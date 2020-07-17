# Deep Learning Based Imbalanced Data Classification for Drug Discovery

Drug discovery studies have become increasingly expensive and time-consuming processes. In the early phase of drug discovery studies, an extensive search has been performed to find drug-like compounds, which then can be optimized over time to become a marketed drug. One of the conventional ways of detecting active compounds is to perform an HTS (high-throughput screening) experiment. As of July 2019, the PubChem repository contains 1.3 million bioassays that are generated through HTS experiments. This feature of PubChem makes it a great resource for performing machine learning algorithms to develop classification models to detect active compounds for drug discovery studies. However, data sets obtained from PubChem are highly imbalanced. This imbalanced nature of the data sets has a negative impact on the classification performance of machine learning algorithms. Here, we explored the classification performance of deep neural networks (DNN) on imbalance compound data sets after applying various data balancing methods. We used five confirmatory HTS bioassays from the PubChem repository and applied one undersampling and three oversampling methods as data balancing methods. We used a fully connected, two-hidden-layer DNN model for the classification of active and inactive molecules.

## Data Sets 

AID485314 is a confirmatory qHTS bioassay for inhibitors of DNA polymerase beta, which is a crucial enzyme for the repair system of human cells.

AID485341 is a confirmatory qHTS bioassay for inhibitors of AmpC beta-lactamase. This assay was carried out to discriminate between aggregators and nonaggregators by not adding detergent. 

AID504466 is a confirmatory qHTS screen for small molecules, which induce genotoxicity in human embryonic kidney cells (HEK293T) expressing luciferase-tagged ELG1.

AID624202 is a confirmatory qHTS bioassay to determine small molecule activators of BRCA1 expression. 

AID651820 is a confirmatory qHTS bioassay for inhibitors of the hepatitis C virus (HCV). The aim here is to identify new
HCV inhibitors against hepatitis C.

## Training, Validation and Test Sets

All training, validation, and test sets through the Kaggle repository. The data sets can be accessed through the following links:

For AID485314: https://www.kaggle.com/selcukorkmaz/pubchemaid485314

For AID485341: https://www.kaggle.com/selcukorkmaz/pubchemaid485341

For AID504466: https://www.kaggle.com/selcukorkmaz/pubchemaid504466

For AID624202: https://www.kaggle.com/selcukorkmaz/pubchemaid624202

For AID651820: https://www.kaggle.com/selcukorkmaz/pubchemaid651820

Paper: Korkmaz, S. (2020). Deep Learning-Based Imbalanced Data Classification for Drug Discovery. Journal of Chemical Information and Modeling. https://dx.doi.org/10.1021/acs.jcim.9b01162
