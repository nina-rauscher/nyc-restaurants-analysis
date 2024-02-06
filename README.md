<h1> NYC Restaurants Analysis </h1>

**POC:** Nina Rauscher (nr2861@columbia.edu)

---

Living in NYC, we are surrounded by restaurants (about 25k!). With such a large offering, we can wonder what distinguishes a thriving restaurant from an average or mediocre one.

This project aims to **identify the key success factors of restaurants in NYC**, in a post-covid era.

To do so, we will start by analyzing the available data to have an overview of the restaurants landscape in NYC. We will then use machine learning techniques to understand the importance of different restaurants features before creating a neural network to predict the satisfaction score of a restaurant based on its characteristics.

<h2> The data </h2>

In order to get data on NYC restaurants, we will use two main data sources:

* **DOHMH New York City Restaurant Inspection Results:** a dataset containing all sanitary inspection results for NYC restaurants along with the restaurants features (e.g. cuisine type, longitude, latitude, phone number)

* **Yelp API:** it will enable us to retrieve a satisfaction score for a restaurant as well as some other characteristics (e.g., reviews, price category...)

We will start by cleaning the data from DOHMH to keep only the latest inspection results for each business and estimate the proportion of restaurants for which the grade is pending to know if it's reasonable to drop it.

Then, we will retrieve the data corresponding to the remaining restaurants using Yelp API phone search to get the business details and id. Once we will have the Yelp ID, we will be able to get an excerpt of the restaurant's reviews by using another API endpoint.

Finally, we will clean this data and save the resulting dataset in a csv file to be used later for EDA and modeling.

<h3> DOMHM New York City Restaurant Inspection Results </h3>

The DOHMH data comes from [this NYC open data page](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) and is updated everyday. The data used in our analysis corresponds to the csv file from Jan 19, 2024.

It contains **27 columns**, including **information on the restaurants' locations** (`ZIPCODE`, `BORO`, `Latitude`, `Longitude`...), **the legal characteristics / logistics** (`CAMIS`: unique restaurant identifier, `DBA`: the name of the restaurant, `PHONE`, `CUISINE DESCRIPTION`, `BIN`...) and **sanitary inspection attributes** (`INSPECTION DATE`, `ACTION`, `SCORE`, `GRADE`...).

The **scarcest columns are `GRADE DATE` and `GRADE`** (a bit less than 50% of the dataset has a value). This is likely due to the fact that our dataset includes all inspection results up to 3 years before the last inspection of the restaurant, and a restaurant can have several sanitary violations counting as several rows.

Most other columns have **less than 2% missing values** so it's fine if we don't focus on them for now.

Initially, there are **212592 rows** and **only 28600 unique restaurants** (identified by their CAMIS). *Wondering how we can have so many rows?* The top 5 CAMIS with the most rows have more than 50 rows associated to them, having had a lot of violations in the past and known a succession of closures/openings 🤯

As explained on *NYC Open Data*, the restaurants with an **inspection date of 1/1/1900 are new establishments that have not yet received an inspection**. Therefore, our next step is to **drop the rows corresponding to these restaurants**.

We only **lose about 7% of the restaurants compared to our initial dataset**, which is pretty decent but we need to do more cleaning to only keep the restaurants with a grade, and take the latest one.

After that, we drop all the rows that don't correspond to the **latest inspection date with non-null and meaningful grade (A, B, C or Z)**, and restrict our dataset to the inspection results **after Jan 1, 2022**. Approximately **80% of the restaurants** included in the original dataset satisfy our criteria.

Then, we only select the columns we are interested in for further analyses (`CAMIS`, `DBA`, `BORO`, `ZIPCODE`, `PHONE`, `CUISINE DESCRIPTION`, `INSPECTION DATE`, `ACTION`, `GRADE`, `SCORE`, `Latitude`, `Longitude`) and drop all rows with a NaN value in one of these columns.

After this cleaning step, we end up with **22561 unique NYC restaurants**, their location, phone number, cuisine and sanitary characteristics. 

In order to use this data as a starting point to query Yelp API and obtain more restaurants attributes, we will need to **reformat the phone number** and **drop phone number duplicates** not to take the risk to associate the wrong ratings to a restaurant.

Eventually, we keep **20,323 restaurants in our dataset**, which is pretty good compared to the 28k initial restaurants and the 24k restaurants with a grade.

<h3> Querying Yelp API to enrich the DOHMH data </h3>

To get more information on our restaurants, we use **Yelp Fusion API** and several API keys to get around the daily limit of 5k API calls.

As we are using 5 API keys, we split our data into 5 dataframes on which we applied a function we created to retrieve Yelp business details based on the phone number. Our function filters to **keep only the restaurants for which we were able to find all the following attributes: `ID`, `NAME`, `RATING`, `PRICE`, `PHONE`, `TRANSACTIONS`, and `REVIEW_COUNT`**.

From our 20k+ restaurants, we only found a bit **less than 9k with all the attributes we wanted on Yelp**. This is pretty disappointing but still **about 1/3 of all NYC restaurants from the DOMHM data**.

We will now get an excerpt of the reviews from these businesses using another API endpoint. Unfortunately, **Yelp API limits the number of reviews we can retrieve to 3 maximum per business**. We will thus use these for EDA but probably not for modeling as it is not necessarily representative of the business as a whole.

Once we have retrieved all the data we wanted from Yelp API, we join our dataframes on the phone number and save the resulting dataset in *'final_restaurants_with_reviews.csv'*.

<h2> Exploratory Data Analysis </h2>

Previously, we retrieved data on NYC restaurants by combining sanitary inspections results (from DOHMH on NYC Open Data) with Yelp business details (rating, reviews excerpt, price category, ...).

Now, we will analyze this data from different standpoints to **get the big picture of the NYC restaurants landscape** and try to **identify some first key success factors**.

**Here are the main axes we will investigate:**
* Statistical analysis
* Geographical visualizations
* Text & sentiment analysis on reviews excerpts

This initial analysis will enable us to **answer questions like**:
* Are the most successful restaurants located in specific boroughs / ZIP codes or are they spread over the 5 boroughs?
* Is there a positive correlation between price and rating?
* Are the restaurants with serious sanitary issues (grade in B, C, Z) perceived as less qualitative by the customers (lower rating on Yelp, more negativity in the reviews)?

<h3> Statistical Analysis </h3>

As expected given how we built the dataset, all ZIP codes correspond to New York City, which means **we don't have geographical outliers**. Moreover, the phone numbers also make sense as they all start with +1...

Besides, the `describe` table gives us some interesting information on our numeric columns:
* **SCORE:** Most restaurants in our dataset have a score below 12 (12 being the threshold of the A grade) and at least 75% of the restaurants have a score below 13. This implies that **most restaurants have a pretty good respect of the sanitary requirements** of New York City. However, there are **a few restaurants with extreme hygiene conditions** as shown by the maximum score of 92!
<br>

* **RATING:** The mean and median rating are very close, around 3.5. **At least 50% of the dataset has a rating between 3.0 and 4.0**. Besides, our 3rd quartile is at 4.0 so it's legitimate to **consider a restaurant as successful when their rating is >= 4.0**. We will use that definition in our upcoming analyses.
<br>

* **REVIEW_COUNT:** There is **a lot of variability** in the review counts for NYC restaurants. **Most of them are below 300** but a significant number of restaurants have a very high review counts (>1k). No value seems weird though as it's very likely to be associated with famous / established restaurants that have been operating for a long time.
<br>

We can now focus on each feature, starting with the sanitary grade and score.

<h4> FOCUS 1 - SANITARY GRADE AND SCORE</h4>

Most restaurants (83+%) have the best grade (A) which is reassuring for our health as customers! About 9% have a B grade (indicating more than minor hygiene violations). The remaining restaurants are whether restaurants with recurring hygiene violations that have been closed and are waiting for the results from a new inspection (Z grade) or restaurants with heavy hygiene violations whose grade is a C.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Distribution of sanitary grades per borough.png" alt="Distribution of sanitary grades per borough" style="width: 70%;">
    </p>
</div>

Besides, as shown by the above chart, **no borough has an alarming sanitary situation** (the non-A grades never represent more than 15-20% of the total population of restaurants).

Now, let's delve into the scores by doing a boxplot:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Boxplot sanitary score.png" alt="Boxplot sanitary score" style="width: 70%;">
    </p>
</div>

As we discovered from the `describe` stats, most of the scores are below 20, with a few restaurants having highest scores (the distribution of these higher scores looks almost uniform between 20 and 60). Thus, **we won't exclude any restaurants due to their scores as we don't notice any abnormal value**.

<h4> FOCUS 2 - RATINGS </h4>

* The mode of the ratings is at 4.0 with about a third of the dataset having this score.
* **85%** of the ratings are comprised **between 3.0 and 4.5**.
* Only less than 1% of the restaurants have an extreme rating (1.0 or 5.0).
* If we consider that a restaurant is successful when it has a **rating >=4.0**, we would have **a bit less than 50%** of our dataset that are successful restaurants. If, however, we have higher standards and only qualify a restaurant as succesful when the **rating is >= 4.5**, we'd only have **about 13%** of restaurants that pass the test.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Distribution of ratings per borough.png" alt="Distribution of ratings per borough" style="width: 70%;">
    </p>
</div>

By plotting the distribution of the ratings per borough, we conclude that:

* As we saw on the previous graphs, **Manhattan has more restaurants** than the other boroughs (not very surprising) but it hosts **slightly fewer best restaurants (5.0) than Brooklyn or the Queens**!
* In our dataset, **no exceptional restaurant (5.0 rating) is located in the Bronx or in Staten Island** and these boroughs are the only one where we can find restaurants with 1.0 average rating.
* Overall, **Manhattan, Brooklyn and Queens are the boroughs with the highest concentration of "successful restaurants"**, regardless of the criteria we adopt (threshold at 4.0 or 4.5).
* **The slightly better sanitary situation of restaurants in the Bronx and Staten Island isn't associated with highest ratings.** However, we cannot conclude that there is no correlation between the sanitary grade and the rating -> we will need to investigate more later!

<br>

Let's now look at the **distribution of successful restaurants per borough** using two potential definitions of successful:
1. Rating >= 4.0
2. Rating >= 4.5

>Note: As we used bokeh for the charts, they might not render on the notebook unless you run the code (including the cell with `output_notebook()`)

**Definition of successful - #1 - Rating >= 4.0**

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Breakdown of successful restaurants by borough 1.png" alt="Breakdown of successful restaurants by borough def 1" style="width: 70%;">
    </p>
</div>

When using 4.0 as a threshold, Brooklyn and Manhattan are the boroughs with the highest proportion of successful restaurants (50%), followed by the Queens (43.5%), Staten Island (37.5%) and the Bronx (33.5%).

**Definition of successful - #2 - Rating >= 4.5**

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Breakdown of successful restaurants by borough 2.png" alt="Breakdown of successful restaurants by borough def 2" style="width: 70%;">
    </p>
</div>

**Once we raise the threshold to 4.5,** the proportion of successful restaurants heavily **decreases** from 30% - 50% to **10% - 15% for each borough**. Thus, the difference between the boroughs is kinda flattened in absolute value but **the max proportion is still equal to approximately 1.5 x the min proportion**.

According to this chart, **Brooklyn is the borough where clients are the most likely to find a very good restaurant (rating>=4.5)** with a proportion of **15.9%**, followed by Manhattan (13.3%), Queens (12.4%), the Bronx (11.2%) and Staten Island (10.8%)

When we will move to modeling, we could thus study two outcomes variable:
* Successful (rating >= 4.0)
* Very successful / Excellent (rating >= 4.5)

<h4> FOCUS 3 - CUISINE x RATING </h4>

Let's look at the **proportion of successful and very successful restaurants per cuisine** 🧑‍🍳

As there are many different cuisines, we will first focus on the top 10 cuisines in terms of ratio between the restaurants satisfying our rating criterion and the total number of restaurants of this cuisine:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Top 10 cuisines ratio successful and excellent.png" alt="Top 10 cuisines ratio successful and excellent" style="width: 90%;">
    </p>
</div>

The **cuisines with the highest proportions** are **healthy (vegan / vegetarian / soups)** or **foreign cuisines (lebanese / australian / haute cuisine)**.

Besides, for most cuisines, the % of excellent restaurants is way below the % of successful / good restaurants.

When looking at the cuisines with the **lowest proportion of successful and excellent restaurants**, we notice that most of them correspond to **fast food / simple food / traditional cuisines** (Donuts, Hamburgers, Tex-Mex) as opposed to the healthy / foreign cuisines of our top 10:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Flop 20 cuisines ratio successful and excellent.png" alt="Flop 20 cuisines ratio successful and excellent" style="width: 90%;">
    </p>
</div>

<h4> FOCUS 4 - PRICE x RATING </h4>

We'd assume that the more expensive a restaurant is, the best the food will be and, therefore, the highest the rating will be. Let's check the validity of this hypothesis:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Distribution of ratings per price category.png" alt="Distribution of ratings per price category" style="width: 70%;">
    </p>
</div>


We notice that our thesis is only partially true. Indeed, we were right about the **absence of very unsatisfying food at expensive restaurants (`$$$` and `$$$$`)** as the lowest rating increases with the price category (`$`: 1.0, `$$`: 2.0, `$$$`:2.5, `$$$$`:3.5). 

**However**, the **extraordinary restaurants (4.5 or 5.0) tend to be associated with cheaper price categories** (`$` or `$$` have the highest number of excellent restaurants and are the only price categories where we find a significant number of 5.0-rated restaurants).


<h4> FOCUS 5 - PRICE x SANITARY SCORE </h4>

Hypothesis: The worst sanitary scores (i.e. the highest) are associated with restaurants in the `$` price category but we won't be able to notice a large difference in scores between the other price categories.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Sanitary inspection score by price category with jittering.png" alt="Sanitary inspection score by price category" style="width: 35%;">
    </p>
</div>

From this scatterplot, we can conclude that **the most expensive restaurants tend to present less serious sanitary violations**. Besides, it appears that **the worst price category in terms of sanitary inspection scores** isn't the cheapest one but **the second cheapest(`$$`)**, probably corresponding to `$10-40`. Thus, we notice a large difference between the two cheapest categories and the two most expensive, with almost all the B and C grades associated with `$` or `$$` restaurants.

<h4> FOCUS 6 - RATING x SANITARY GRADE </h4>

We're interested in visualizing the distribution of sanitary grades across restaurants ratings. Intuitively, we could think that a cleaner restaurant leads to a higher satisfaction, let's check if that's true!

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Distribution of sanitary grades per rating category.png" alt="Distribution of sanitary grades per rating category" style="width: 70%;">
    </p>
</div>

What we see is quite surprising: 
- All the extremely bad restaurants (<=2.0) have the best grade (A)
- Most restaurants with serious sanitary issues would be considered as decent or good (3.5 or 4.0)

Thus, **the sanitary grade doesn't seem to be a great predictor of the Yelp rating**...

<h4> FOCUS 7 - TRANSACTIONS x RATING </h4>

The `TRANSACTIONS` attribute corresponds to a list of strings, each string being an option of accessing the restaurant food.
There are 3 options (non mutually exclusive):
- `pickup`
- `delivery`
- `restaurant_reservation`

From the `value_counts()`, we can tell that **most of our restaurants offer both delivery and pickup, or just delivery**. We also have **a significant number of restaurants that have no options** on their Yelp profile.

After reformatting the data, we can plot the distribution of transactions category per rating:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Distribution of transactions category per rating.png" alt="Distribution of transactions category per rating" style="width: 70%;">
    </p>
</div>

* The highest ratings mainly offer delivery and pickup, just delivery or nothing.
* The restaurants that offer restaurant reservations tend to be associated with higher ratings (4.0 or 4.5).
* Overall, this field doesn't seem to be very consistent as it seems unrealistic that most restaurants don't offer reservation...

<h4> FOCUS 8 - REVIEW COUNT x RATING </h4>

One of the last restaurant features we've not looked at is the review count. Let's do a scatter plot with jittering!

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Review count by rating with jittering.png" alt="Review count by rating with jittering" style="width: 40%;">
    </p>
</div>

Surprisingly, the excellent restaurants (rating of 5.0) almost all have very few ratings (<400) except one with about 2k reviews. The restaurants with a very high number of reviews tend to be pretty good or decent (3.5 - 4.5). Having a high number of ratings (>1k) would thus indicate that the restaurants have at least a decent quality but the reciprocal isn't true.

<h4> FOCUS 9 - ZIPCODE x RATING </h4>

We will now determine which ZIP codes have the more successful (rating>=4.0) and excellent restaurants (rating>=4.5). 

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Top 10 zipcodes number.png" alt="Top 10 ZIP codes number" style="width: 90%;">
    </p>
</div>

According to these charts, **the best ZIP codes seem to be 10002, 10003 and 11211**. 

Let's see how these rankings evolve when looking at the **proportion of all restaurants** in these ZIP codes that have a >= 4.0 and >= 4.5 rating. It is important to ensure that we're not just seeing a lot of very good restaurants in these ZIP codes because of a high number of restaurants.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Top 10 zipcodes proportion.png" alt="Top 10 ZIP codes proportion" style="width: 90%;">
    </p>
</div>

Except **11211 (corresponding to Williamsburg)**, all the top ZIP codes from our previous rankings aren't in the top 10 by proportion. This means that they appear in the top 10 by count because they have a lot of restaurants and out of all these restaurants, a significant number has a good rating. 

However, the best ZIP codes by proportion are likely to be associated with areas where there aren't a lot of restaurants, making it easier to reach 100% of restaurants with average rating >=4.0 or 4.5.

Besides, we notice that even for the best ZIP codes, **it's hard to maintain a proportion of excellent restaurants above 1/3** (only 3 ZIP codes meet that criteria).

<h3> Geo Visualizations </h3>

Ratings and other restaurants attributes vary a lot depending on the location of the business. Thus, plotting our data on a map of New York City will enable us to better visualize our findings.

<h4> MAP 1: Heatmap of the density of restaurants in each NYC ZIP code </h4>

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Heatmap of the density of restaurants in each NYC ZIP code.png" alt="Heatmap of the density of restaurants in each NYC ZIP code" style="width: 90%;">
    </p>
</div>

As we noticed when looking at the distribution of the number of restaurants by borough, **Manhattan dominates**, especially the **ZIP codes below Central Park**. **Brooklyn's Dumbo, Williamsburg , Park Slope, Sunset Park and Bay Ridge** also reach a **high density of restaurants**, as explained by the dense activity and tourism. Other neighborhoods are more residential and, therefore, host less restaurants.

<h4> MAP 2: Heatmap of the number of successful restaurants in each NYC ZIP code </h4>

Let's now take a look at the heatmap of good restaurants based on Yelp rating:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Heatmap of the number of successful restaurants in NYC.png" alt="Heatmap of the number of successful restaurants in NYC" style="width: 90%;">
    </p>
</div>

Compared to our previous heatmap, we notice some new black ZIP codes which correspond to ZIP codes with no restaurants with a Yelp rating greater than or equal to 4.0.

Besides, some neighborhoods improved their "performance", including **Williamsburg, K-town, the West Village and the Lower East Side**. This means that **even though they don't have as many restaurants as SoHo/TriBeCa, they have almost the same number of good restaurants**.


<h4> MAP 3: Heatmap of the number of excellent restaurants in each NYC ZIP code </h4>

Now, let's raise the bar and only display the number of restaurants with a rating greater than or equal to 4.5:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Heatmap of the number of excellent restaurants in NYC.png" alt="Heatmap of the number of excellent restaurants in NYC" style="width: 90%;">
    </p>
</div>

Interestingly, **most of the great neighborhoods for excellent restaurants aren't the neighborhoods where you find a lot of restaurants** but the neighborhoods with a **moderate density of restaurants (Park Slope, Williamsburg, Lowe East Side)**.

Overall, **the Queens, the North of Brooklyn and the South of Manhattan (excluding FiDi and up to Central Park)** are the places where you'll find the **best restaurants** according to Yelp ratings ✨

<h4> MAP 4: Heatmap of the number of very bad restaurants in each NYC ZIP code </h4>

We will define a very bad restaurant as a restaurant with a Yelp rating <= 2.0.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Heatmap of the number of very bad restaurants in NYC.png" alt="Heatmap of the number of very bad restaurants in NYC" style="width: 90%;">
    </p>
</div>

Surprisingly, despite having a huge number of restaurants, Manhattan isn't the borough hosting most of the worst restaurants. 

The ZIP codes associated with a lot of very bad restaurants aren't ZIP codes that have many restaurants. 

**The proportion of restaurants with very bad ratings can reach about 50% in some ZIP codes in the Bronx, Queens or East Brooklyn (Carnasie, Buschwick, Glendale)**, not great for the local residents who don't want to travel to get decent food...

<h3> Text Mining - Wordclouds </h3>

As we got a few reviews (between 1 and 3) for all restaurants in our dataset, let's visualize the recurring words by drawing wordclouds!

First, we leveraged NLTK and gensim to build a function taking a column and a value as parameters and creating a wordcloud of the reviews of these restaurants as output.

Then, we applied it to each value of price and grade.

<h4> WORDCLOUDS - PRICE </h4>

The wordcloud associated with the `$` price category is as follows:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Wordcloud one dollar.png" alt="Wordcloud one dollar" style="width: 50%;">
    </p>
</div>

The cheapest restaurants are mainly associated with fast food ("pizza", "chicken", "quick", "sandwich"). Overall, the words aren't negative ('amazing', 'great', 'friendly').

For the `$$` price category:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Wordcloud two dollars.png" alt="Wordcloud two dollars" style="width: 50%;">
    </p>
</div>

The words are pretty similar to what we got with the cheapest restaurants though we notice the apparition of terms referring to the evening ('night', 'dinner') and new kind of food ('sushi').

Now, regarding the `$$$` price category:

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Wordcloud three dollars.png" alt="Wordcloud three dollars" style="width: 50%;">
    </p>
</div>

Following the trend of the previous price category ('$$'), this one seems to be dedicated to dinner experience ('dinner', 'night', 'experience', 'reservation', 'party', 'ambiance', 'atmosphere') more than a restaurant where you go grab lunch a bit randomly.

Finally, the most expensive price category (`$$$$`):

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Wordcloud four dollars.png" alt="Wordcloud four dollars" style="width: 50%;">
    </p>
</div>

As expected, the priciest category is even more associated to dining and tasting. Besides, it appears that many extremely expensive restaurants in NYC are omakase / sushi restaurants or premium steakhouses.

<h4> WORDCLOUDS - SANITARY GRADE </h4>

These ones are quite disappointing as they aren't very different one from another. It makes sense though when we recall that most restaurants with B/C/Z aren't very badly rated (almost all >= 3.0). Thus, it seems to confirm that customers don't notice the sanitary violations happening at these restaurants and so make an opinion independently from the sanitary situation (i.e. grade).

<h3> Sentiment Analysis </h3>

We will use the *NRC-Emotion-Lexicon-Wordlevel-v0.92* to get the emotions associated with each word, and build a function `comparative_emotion_analyzer` that creates a dataframe with scores for each element (usually concatenated reviews), for the following emotions: *Fear,	Trust,	Negative,	Positive,	Joy,	Disgust,	Anticipation,	Sadness,*	and *Surprise*.

However, as Yelp API didn't enable us to retrieve the whole reviews, our concatenated reviews are weird and not fully representative of the restaurants (only  3 reviews per business maximum and these are often truncated). Using these emotions in the upcoming machine learning models would lead to potential issues as the data isn't representative of the restaurants. Besides, knowing all the reviews is almost like knowing the rating.

Let's just look at the emotion analysis per cuisine, borough, rating and price category for fun!

<h4> FOCUS 1: EMOTION ANALYSIS PER CUISINE </h4>

The difference between positive and negative could be interesting to look at and in some cases (e.g., Soups, Not Listed/Not Applicable) it confirms our previous findings. However, there are also cuisines for which we'd have expected a lower value of Positive - Negative (e.g. Hamburgers, Sandwiches) given their low proportion of restaurants with a rating >= 4.0.

<h4> FOCUS 2: EMOTION ANALYSIS PER BOROUGH </h4>

All boroughs have pretty similar scores so we can't conclude many things from these.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Emotion analysis per borough.png" alt="Emotion analysis per borough" style="width: 70%;">
    </p>
</div>

<h4> FOCUS 3: EMOTION ANALYSIS PER PRICE </h4>

We mainly notice an improvement between `$` and `$$$$` (Positive - Negative going from 4.5 to 6). The two middle categories are quite similar to each other, better than `$` and slightly below `$$$$`).

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Emotion analysis per price.png" alt="Emotion analysis per price" style="width: 70%;">
    </p>
</div>

<h4> FOCUS 4: EMOTION ANALYSIS PER RATING </h4>

As expected, the higher the rating, the lower the negative score and the higher the positive score. This is reassuring on the fact that our reviews excerpt are still okay even though not great due to the API limits.

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Emotion analysis per rating.png" alt="Emotion analysis per rating" style="width: 70%;">
    </p>
</div>

<h3> Topic Modeling </h3>

Although the number of reviews available is very small, we will write the code for topic modeling and look at the results but it's unlikely to be very satisfying as the excerpt are not long enough to identify meaningful topics.

Our first step is to preprocess and vectorize the reviews. We've decided to **include bigrams** if they occur in at least 20 reviews.

We end up having **2495 unique tokens**, which is **pretty low** compared to our number of documents (8975 concatenated reviews). This can be explained by the small size of each review excerpt (limited to 3 reviews by Yelp API and the reviews are truncated for a not yet explained reason...). We will proceed but aren't expecting great results 😕

Thanks to this coherence score (u mass) chart, we determine that the best number of topics, i.e. the one minimizing coherence, is 3. 

<div style="flex: 48%;">
    <p align="center">
      <img src="00 - Images/Evolution of the average coherence score.png" alt="Evolution of the average coherence score" style="width: 70%;">
    </p>
</div>

This is the number we intuitively chose given the number of unique tokens we had. Unfortunately, training the LDA model with this number of topics, we get topics that are quite hard to interpret:

* **Topic A:** ([(0.01662321, 'they'),
   (0.011962658, 'you'),
   (0.011440827, 'but'),
   (0.010377562, 'not'),
   (0.010264887, 'order'),
   (0.0100071225, 'have'),
   (0.009859836, 'pizza'),
   (0.009838225, 'on'),
   (0.009566593, 'that'),
   (0.0094753215, 'place'),
   (0.009006042, 'here'),
   (0.008362386, 'time'),
   (0.008184554, 'so'),
   (0.008050041, 'are'),
   (0.007598232, 'with'),
   (0.007520197, 'service'),
   (0.007172889, 'me'),
   (0.0069971, 'at'),
   (0.0069168955, 'from'),
   (0.006790867, 'good')],
  -1.1169050325858165)

* **Topic B:** ([(0.023003161, 'we'),
   (0.01505546, 'with'),
   (0.014180245, 'place'),
   (0.013538716, 'on'),
   (0.01310399, 'here'),
   (0.012033439, 'great'),
   (0.010407596, 'had'),
   (0.010322808, 'were'),
   (0.009863686, 'but'),
   (0.009360499, 'at'),
   (0.009317959, 'service'),
   (0.009044395, 'came'),
   (0.008643195, 'good'),
   (0.00796429, 'so'),
   (0.0073114615, 'very'),
   (0.007219161, 'restaurant'),
   (0.007056253, 'that'),
   (0.006444141, 'have'),
   (0.0064103375, 'you'),
   (0.006326039, 'this_place')],
  -1.1284815116895972)

* **Topic C:** ([(0.016107643, 'good'),
   (0.014449326, 'with'),
   (0.013787638, 'chicken'),
   (0.013005145, 'place'),
   (0.010204024, 'ordered'),
   (0.01002934, 'are'),
   (0.0093176775, 'but'),
   (0.009263302, 'great'),
   (0.008635818, 'they'),
   (0.008555256, 'so'),
   (0.008505526, 'delicious'),
   (0.0084666805, 'their'),
   (0.007923528, 'very'),
   (0.007489106, 'spot'),
   (0.007455237, 'had'),
   (0.007355466, 'chinese'),
   (0.0069447593, 'here'),
   (0.0066659376, 'rice'),
   (0.006577057, 'not'),
   (0.0064564375, 'this_place')],
  -1.410515712481643)

To improve the resulting topics, we could have added more stopwords but the main improvement would come from longer reviews and the disparition of the limit to 3 reviews per business on Yelp API.

<h3> Insights Summary </h3>

Let's briefly summarize our main findings:

* **Restaurants aren't uniformly distributed** across New York City: Manhattan hosts most restaurants, followed by Brooklyn, the Queens, Bronx and Staten Island.

* When qualifying a restaurant as successful when its **Yelp rating is >= 4.0**, **Brooklyn and Manhattan** are the boroughs with the **highest proportion of successful restaurants (50%)**, followed by the Queens (43.5%), Staten Island (37.5%) and the Bronx (33.5%).

* **Once we raise the threshold to 4.5,** the proportion of successful restaurants heavily **decreases** from 30% - 50% to **10% - 15% for each borough**. **Brooklyn is the borough where clients are the most likely to find a very good restaurant (rating>=4.5)** with a proportion of **15.9%**, followed by Manhattan (13.3%), Queens (12.4%), the Bronx (11.2%) and Staten Island (10.8%).

* **Fast food and traditional cuisines** ('Donuts', 'Hamburgers', 'Pancakes/Waffles', 'Soups/Salads/Sandwiches') **have very low ratings** while most **foreign cuisines and healthy food** tend to have a **higher ratings**.

* Despite the **absence of very unsatisfying food at expensive restaurants** (`$$$` or `$$$$`), the **extraordinary restaurants (4.5 or 5.0) are more likely to be associated with cheaper price categories** (`$` or `$$` have the highest number of excellent restaurants and are the only price categories where we find a significant number of 5.0-rated restaurants).

* Surprisingly, **the sanitary grade doesn't seem to be a great predictor of the Yelp rating**... Indeed, all the extremely bad restaurants (<=2.0) have the best grade (A) and most restaurants with serious sanitary issues would be considered as decent or good (3.5 or 4.0).

* However, **the sanitary score is logically better (i.e., lower) for the restaurants with an expensive menu** (`$$$` or `$$$$`). Thus, we notice a **strong difference** in terms of sanitary violations between the two cheapest categories and the two most expensive, with **almost all the B and C grades associated with `$` or `$$` restaurants.**

* Regarding Yelp transactions, **the highest ratings mainly offer delivery and pickup, just delivery or nothing**. In addition, the restaurants that offer **restaurant reservations** tend to be associated with **higher ratings** (4.0 or 4.5). **Nevertheless, this field doesn't seem to be super reliable** as there are very few restaurants with the reservation option in their Yelp transactions.

* The best neighborhoods in terms of number of restaurants with an **average rating above 4.5** are the **Queens**, **the North of Brooklyn** (**Williamsburg, Park Slope**) and the **South of Manhattan** (excluding FiDi and up to Central Park).

* The proportion of restaurants with **very bad ratings** can reach **about 50%** in some ZIP codes in **the Bronx, Queens or East Brooklyn (Carnasie, Buschwick, Glendale)**.

* Eventually, we won't use emotion scores for modeling as the reviews are too limited by Yelp API.

<h2> Machine Learning </h2>

WIP
