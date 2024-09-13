### Dataset Description
For this competition we have generated a dataset of outcomes from different variants of Monte-Carlo tree search (MCTS) agents playing over a thousand distinct board games. All of the games are two-player, sequential, zero-sum board games with perfect information. Your task is to predict the degree of advantage the first agent has over the other.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data and sample submission will be provided to your notebook in batches of 100 rows via an evaluation API.

### Files
#### train.csv
Every row of data represents a set of plays between an ordered pair of two specific agents, in a single game, with a single outcome per play.

* **Id** - (integer) A unique (within this file) ID for this row of data. The test data also has an Id column, which (like the one for training data) also starts counting at 0, but these are unrelated to each other. You should probably drop this column.
* **GameRulesetName** - (string) A combination of the game name and ruleset name in Ludii. Within the Ludii system, there is a distinction between games (cultural artifacts) and rulesets (the same game might be played according to different rulesets). For the purposes of this competition, think of every unique combination of a game + a ruleset as a separate game, although some games (ones that have many different rulesets) might be considered overrepresented in the training data.
* **agent[1/2]** - (string) A string description of the agent that played as the [first/second] player. See the section on Agent String Descriptions below for more info.
* ... - Most remaining columns describe properties of the game + ruleset played in this row. These range from abstract features (is the game deterministic or stochastic?) to specific features (does the game use a star-shaped board?), and from features that are about how the rules are described (e.g., do any of the rules involve a greater-than comparison?) to features that are about the behavior of the game in practice (e.g., number of times per second we can run a completely random play-out from initial game state till the end on our hardware). For more details see the Concepts page on the Ludii website, and the publication about them, or concepts.csv.
* **EnglishRules** - (string) An natural language (English) description of the rules of the game. This description is not guaranteed to be self-contained (e.g., it might refer to rules from famous other games, such as Chess, for brevity), unambiguous, or perfectly complete.
* **LudRules** - (string) The description of the game in Ludii's game description language. This is the description that was used to compile the game inside Ludii and run the simulations, so it is always guaranteed to be 100% complete and unambiguous. However, this is a formal language that most existing Large Language Models / foundation models have likely received little, if any, exposure to.
* **num_[wins/draws/losses]_agent1** - (int) The number of times the first agent [won/drew/lost] against the second agent in this game and this specific matchup of two agents.
* **utility_agent1** - (float) The target column. The utility value that the first agent received, aggregated over all simulations we ran for this specific pair of agents in this game. This value will be between -1 (if the first agent lost every single game) and 1 (if the first agent won every single game). Utility is calculated as (n_games_won - n_games_lost) / n_games.

#### test.csv
The same as train.csv minus the following columns: num_wins_agent1, num_draws_agent1, num_losses_agent1, and utility_agent1 columns. Expect approximately 60,000 rows in the hidden test set.

#### sample_submission.csv
An example valid submission file. Note that the evaluation API will generate the final submission.

* **Id**
* **utility_agent1** - (float) These should be your predictions, which should be between -1.0 and 1.0 (both inclusive).

#### concepts.csv
A file, exported from the publicly available Ludii database, containing information on the Concept-based features of games. This may be useful for filtering out certain categories of features. For example, as concepts of the "Visual" category should in principle not have any effect on AI playing strength you may wish to drop concepts with a TypeId of 9 (although there may be correlations between types of games and how humans choose to visualise them, so you may also choose not to do this).

* **Id** - (int) A unique ID for every concept.
* **Name** - (string) A unique name for every concept (this shows up as column name in the training and test data).
* **Description** - (string) A brief natural language description of what the concept is about.
* **TypeId** - (int) The ID of the type of concept, as per the ConceptType enum in Ludii's source code.
* **DataTypeId** - (int) An ID for the data type of this concept; 1 for Boolean, 2 for Integer, 3 for String, 4 for Double.
* **ComputationTypeId** - (int) An ID for the "computation type" of the concept; 1 if it is a concept of which the value may be determined purely from compiling the game (e.g., is this a stochastic game?), or 2 if it is a concept of which the value must be determined by running simulations (e.g., how many tuns does this game last on average?). For the latter type, we used play-outs between agents that select moves uniformly at random to compute the concept values.
* **TaxonomyString** - (string) A description of where the concept is located in the hierarchical taxonomy of all concepts. This is visible on the Ludii website.
* **LeafNode** - (bool) 0 or 1 depending on whether or not this concept is a leaf node of the taxonomy.
* **ShowOnWebsite** - (bool) 0 or 1 depending on whether or not this concept is shown on the Ludii website.

### Agent String Descriptions
All agent string descriptions in training and test data are in the following format: MCTS-<SELECTION>-<EXPLORATION_CONST>-<PLAYOUT>-<SCORE_BOUNDS>, where:

* **<SELECTION>** is one of: UCB1, UCB1GRAVE, ProgressiveHistory, UCB1Tuned. These are different strategies that may be used within the Selection phase of the MCTS algorithm.
* **EXPLORATION_CONST** is one of: 0.1, 0.6, 1.41421356237. These are three different values that we have tested for the "exploration constant" (a numeric hyperparameter shared among all of the tested Selection strategies).
* **PLAYOUT** is one of: Random200, MAST, NST. These are different strategies that may be used within the Play-out phase of the MCTS algorithm.
* **SCORE_BOUNDS** is one of: true or false, indicating whether or not a "Score-Bounded" version of MCTS (a version of the algorithm that can prove when certain nodes in its search tree are wins/losses/draws under perfect play, and adapt the search accordingly).
For example, an MCTS agent that uses the UCB1GRAVE selection strategy, an exploration constant of 0.1, the NST play-out strategy, and Score Bounds, will be described as MCTS-UCB1GRAVE-0.1-NST-true.

You may treat every distinct agent string as a completely separate agent, but you may also try to leverage the compositional nature of the MCTS agents and split it up into its components.
