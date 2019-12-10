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

### Results for Defect Builds on Ascension 20:

**cluster_0: Dark Expanse**  
At 153 resources, this is nearly triple the size of any of the other builds. It is also the only build to have resource values of <1, which means that it's more flexible. As far as relics go, all the 'required' relics (with a value of 1) are among the highest rated, with the exception of 'Symbiotic Virus', which has a dark-orb focus. Adding to this, of the 11 required cards, one is 'Darkness', which synergizes with Symbiotic Virus as well as 'Doom and Gloom', which is listed, but not required. The existence of 'Aggregate' (a card which provides energy based on the number in your draw pile) and 'Apotheosis' (a card which upgrades all cards in your deck for that combat) adds the notion that this build is meant to have the largest deck size. 'Compile Driver' is also a required card (provides card draw based on the number of unique orbs) and it goes with the inclusion of other orb-generating cards such as 'Glacier' and 'Capacitor'. Overall, I think the most profound features of this build are the emphasis on deck size, dark-focus, and variety of orbs.

**cluster_1: Any Color You Like**  
There are 60 resources in total with 31 relics and 29 cards. Most of the relics are just the generic highly-rated ones, but the inclusion of both 'Inserter' and 'Data Disk' gives this build the flavor of an orb and focus-build. As far as cards go, there are 'Chaos', 'Cold Snap', 'Coolheaded', 'Defragment', 'Fission', 'Fusion', 'Meteor Strike', and 'Rainbow' to give the strong impression of an orb-focus.

**cluster_2: Power Overwhelming**  
This build includes 51 resources—19 relics and 32 cards. Aside from the usual high-rated relics, included are 'Inserter', 'Symbiotic Virus', and 'Nuclear Battery'. This points to another orb and focus build, but with an emphasis on energy. Supporting this, 'Biased Cognition', 'Conserve Battery', 'Consume', 'Defragment', 'Electrodynamics', 'Meteor Strike', and 'Tempest', and 'Thunderstrike' were all included cards. It looks like the main synergy is in using the high-energy for channeling lightning and fusion.

**cluster_3: Curses**  
Another build at 51 cards, this time with 21 relics and 30 cards. The relics that stand out are 'Cracked Core', 'Dead Branch', 'Du-Vu Doll', and 'Peace Pipe'. This gives the impression of a curse/strength build since 'Du-Vu Doll' adds strength for every curse in your deck, 'Cracked Core' adds curses, and 'Dead Branch' allows you to draw cards when curses are exhausted. On the card side, we see 'Clumsy', 'Doubt', 'Parasite', and 'Write'—all curses.

**cluster_4: Get Paid**  
Coming in at 66 resources, this build is heavy on the relics with 43 and only 24 cards. Why the imbalance? It looks like this build is focused on buying relics at every chance from merchants. To this end, 'Maw Bank', 'Membership Card', 'Smiling Mask', 'Lee's Waffle', and 'Chemical X' are all present and either help with buying relics or can only be purchased from the merchant. The cards are a fairly generic mix of highly-rated cards with the exception of 'Hand of Greed', which is a colorless card that earns the player money in combat.

**cluster_5: Everything Must Go**  
At 73 resources, this is the second largest build with 46 relics and 27 cards. The relics are mostly just high-rate, but what stood out is 'Spinning Top'—a relic that gives a free card draw if you have no cards in your hand. This typically pairs well with 0-cost cards sinece you could theoretically keep playing infinitely if all your cards cost 0. To this end, 'Anger', 'Calculated Gamble', 'Chrysalis', 'Concentrate', 'Double Energy', 'Enlightenment', 'Escape Plan', 'Finesse', 'Claw' (referered to as 'Gash' historically), 'Havoc', 'Jack of all Trades', and others all help toward this end. At first, I was a bit confused by the cards listed since an inordinate number are from non-Defect card lists. This is only possible with the relic 'Prismatic Shard'. My hunch is that this became its own cluster since anyone getting 'Prismatic Shard' would be the only ones with these cards, hence there being a lot of distance to the other nodes.

**cluster_6: Come at Me**  
This build has 49 resources with 30 relics and 19 cards. It is strongly focused on passively doing damage while maintaining huge amounts of block. Relics include 'Bronze Scales', 'Cables', 'Fossilized Helix', 'Incense Burner', 'Letter Opener', 'Mercury Hourglass', and even the lower-rated 'Runic Dome' since enemy intent doesn't matter when you're blocking so well. The cards are strangely non-Defect oriented, implying a Prismatic Shard, but focused on blocking and doing passive damage. They include 'Backflip', 'Blur', 'Coolheaded', 'Finesse', 'Escape Plan', 'Juggernaut', 'Loop', and 'Panache'.

**cluster_7: Modded**  
At 47 resources, this is the smallest build and it favors cards with 21 relics and 25 cards. It's also the only one with modded cards. In particular, I'm seeing cards from Replay the Spire. This can be attributed to my inability to remove these cards effectively, given no clear signal on which to subset these out. This being the case, it's hard for me to really describe the synergies of the build since I don't know what 'Blue Doll', 'Arrowhead', 'Replay Funnel', and other modded cards are. On the bright side, the fact that they're all together suggests the clustering algorithm did a good job.

**cluster_8: Claw-Finale**  
This build has 62 resources with a huge majority for relics at 46 with only 16 cards. The reason for the scarcity of cards is that the focus is on having an extremely thin deck and playing as many Claws (called 'Gash' in the data) as possible. This is another build which has non-Defect cards and it creates the opportunity for an extreme synergy: playing Burst to play the next skill twice, then playing Dual Weild (twice) to add a total of 4 Claws to your hand. Paired with 'Master of Strategy', 'Prepared', 'Go for the Eyes', 'True Grit', and, most importantly, 'Grand Finale', the deck can be manipulated easily to make 'Grand Finale' land as often as possible as long as there are few cards overall.

