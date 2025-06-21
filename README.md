# IMDb Movie Analysis and Genre-Based Rating Prediction

## Objective:
#### The primary goal of this project is to explore and analyze the top 1000 movies listed on IMDb, identify key trends and patterns in genres, ratings, and financial figures, and build a simple machine learning model to predict IMDb ratings based on movie genres. This project combines data preprocessing, visualization, and linear regression to generate insights and predictive analytics from movie data.

## Dataset:
#### The analysis is conducted on the IMDb Top 1000 Movies dataset. It includes features such as title, genre, runtime, release year, budget, gross revenue, director, actors, and IMDb rating.

## Key Steps and Methodology:

### Data Preprocessing and Cleaning:

#### - Extracted numerical values from text-based fields like runtime, budget, and gross revenue.

#### - Derived new features such as Main_Genre, Main_Director, and Num_Actors.

#### - Handled missing values using appropriate imputation strategies (e.g., replacing missing directors with "Unknown").

#### - Converted date and financial columns into suitable numerical formats.

### Exploratory Data Analysis (EDA):

#### - Visualized the distribution of movie genres and release years.

#### - Analyzed the average IMDb rating per genre.

#### - Explored the relationship between IMDb ratings and gross income using scatter plots.

#### - Calculated and visualized the correlation between numerical features using a heatmap.

### Machine Learning - IMDb Rating Prediction:

#### - Built a linear regression model to predict IMDb ratings using the main movie genre as the sole predictor.

#### - Applied one-hot encoding to transform categorical genres into numerical format.

#### - Split the dataset into training and test sets (80/20).

#### - Evaluated the model using RMSE and R² score.

### Results:

#### The model provides a basic estimation of IMDb ratings based solely on genre.

### Final performance metrics included:

#### - Root Mean Squared Error (RMSE): ~0.53

#### - R² Score: ~0.18

## Conclusion:
#### While the genre is a contributing factor to a movie's IMDb rating, it alone does not provide a strong predictive power (as reflected by the low R² score). Nevertheless, the project successfully demonstrates the complete pipeline of data cleaning, analysis, and modeling, and lays the groundwork for more complex models using additional features.
