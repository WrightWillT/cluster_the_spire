# cluster_the_spire
Clustering victorious Spirelogs games into distinct game-winning decks

### Context and Purpose

**Context:** Spirelogs' creator, Alleji, graciously gave me access to the nearly 280K games that have been uploaded to the site.  This data provides all the game-features and turn indices to recreate an entire game, including the final deck and set of relics. It also includes features to help subset, including Ascension level, character, and win condition.  Initially, my intent was to borrow an existing algorithm to cluster decks and relics together into winning combinations of Ascension 20 wins for Defect.  However, after looking through the documentation, the closest algorithm I could find was [mean shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift), but it measures distance differently.  This being the case, my goal shifted to solving this problem.

**Purpose:** The primary aim of this project is to create a machine learning algorithm for unsupervised clustering and classification of binary sequences. The secondary aim is to apply this algorithm for one of the three characters (Defect) for victorious Ascension 20 runs. The tertiary aim (post-deadline for the project) will be to complete the clustering for all characters.

Once complete, all results will be moved into Google Sheets and shared with various Slay the Spire forums to help interested players make better decisions and get better at the game.

### What's included

In the root folder:
- All the folders described below
- The 'A5_Final_Project_plan.ipynb' script, which outlines the original intent for the project and the details about where the data came from.
- A 'LICENSE' file, which is an open MIT license
- A single 'example_run.json' file to check out the structure of the data within the files (mainly intended for folks who want to see if this dataset would be useful for their own investigation)
- This README

Within the 'src' folder: 
- 'cluster_the_spire.ipynb' a jupyter notebook which does it all.  It unzips the raw data, cleans it, subsets it, generates the resource frequency tables per character, and clusters the results
- 'clustering_algorithm_methodology_testing.ipynb' is where I developed the a machine learning algorithm for unsupervised clustering of binary sequences.  It is somewhat similar to the mean shift algorithm, but it computes distance in a non-euclidian way and takes an iterative approach starting at k=m (where m is the number of observations) through k=2.  It also allows for a weighted 'tolerance' which essentially takes large steps, which gradually get smaller as k approaches 2.  Included in this script is all the nitty-gritty details about how it works along with tests on simulated data, which resembles the Spirelogs data.

Within the 'data_raw folder': ~280K Slay the Spire game runs in both a zipped and unzipped format. Only the zipped version is needed for the analysis since the cluster_the_spire.ipynb script unzips it, but since it's faster to start with the unzipped data, that version has been included for convenience. The unzipped data is bucketed into date-based folders, indicating when the data was uploaded to Spirelogs.  All data is in JSON format.

Within the 'results' folder: at the time of writing this, there are only character-specific resource frequency tables in .csvs.  My intention is to augment these tables to include columns for each cluster with values for the percent of games within the cluster which contain the resource.

### Limitations
The algorithm works fantastically on small simulated data, but takes FOREVER to run on the full-featured data due to the volume of distance calculations that must be performed. This being the case, I may end up submitting the results of a subset of the data so that it's done before the project deadline (I'm cutting it close).  

