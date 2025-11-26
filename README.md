Loaded books.csv and reviews.csv in pandas.

First filtered books.csv with Rating 5.0 and rating number 1234(using reviews.csv)
Then from those filtered books, search for those whose isbn10 column is from that review whose parent_asin is the given hash computed(STU011- > sha256 -> first 8 characters.)

then after finding the book, filtered all reviews of that book.->89 reviews found for the book.

Built a text classifier using weak labeling rules to separate suspicious vs genuine reviews. Used SHAP to identify the top 3 authenticity-indicative words and computed FLAG3.
