# ragsearch

`ragsearch` is a Python library designed for building a Retrieval-Augmented Generation (RAG) application that enables natural language querying over structured data. This tool leverages embedding models and a vector database (FAISS) to provide an efficient and scalable search engine.

## Features
- Seamless integration with the Cohere AI LLM for generating embeddings.
- Utilizes FAISS for fast, in-memory vector storage and similarity search.
- Easy setup and configuration for different use cases.
- Simple web interface for user interaction.

## Installation
To install `ragsearch`, run the following command:

```bash
pip install ragsearch
```

Alternatively, for local development, use:

```bash
pip install /path/to/ragsearch
```

Ensure that you have all necessary dependencies installed:

```bash
pip install pandas faiss-cpu flask cohere
```

## Basic Setup
### Step 1: Prepare Your Data
Ensure you have your structured data in a CSV, JSON, or Parquet format. The data should include columns for the main content and any relevant metadata.

**Example data (`sample_data.csv`)**:
```csv
name,id,minutes,contributor_id,submitted,tags,n_steps,steps,description,ingredients,n_ingredients,average_rating,votes,Score,calories,total fat (PDV),sugar (PDV),sodium (PDV),protein (PDV),saturated fat (PDV),carbohydrates (PDV),category,meal_type,cuisine,difficulty
baked ham glazed with pineapple and chipotle peppers,146558,85,58104,2005-11-28,"['ham', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'lunch', 'main-dish', 'pork', 'american', 'mexican', 'southwestern-united-states', 'tex-mex', 'oven', 'holiday-event', 'easter', 'stove-top', 'spicy', 'christmas', 'meat', 'taste-mood', 'sweet', 'equipment', 'presentation', 'served-hot', '4-hours-or-less']",7,"['mix cornstarch with a little cold water to dissolve', 'place all ingredients except for ham in a blender and blend smooth , in a small saucepan over medium heat bring to a boil them simmer till thickened', 'preheat oven to 375 f', 'place ham , cut end down , in a large baking pan and score skin', 'bake ham for 15 minutes', 'brush glaze over ham and bake for another hour or until internal temperature reads 140 f', 'baste half way through baking']","sweet, smokey and spicy! go ahead and leave the seeds in if you enjoy the heat.","['smoked ham', 'brown sugar', 'crushed pineapple', 'chipotle chile in adobo', 'adobo sauce', 'nutmeg', 'fresh ginger', 'cornstarch', 'salt']",9,5.0,27,4.852754009963201,712.5,50.0,127.0,207.0,131.0,55.0,12.0,Non-veg,Lunch,North-American,2.65
chocolate raspberry  or strawberry  tall cake,90774,60,117781,2004-05-05,"['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'occasion', 'desserts', 'oven', 'dinner-party', 'holiday-event', 'cakes', 'chocolate', 'dietary', 'equipment']",20,"['prepare the cake according to package directions , using three greased and floured 9 in round cake pans', 'bake at 350 degrees for 25- 30 minutes or until a toothpick inserted in center comes out clean', 'cool for 10 minutes', 'remove from pans to wire racks to cool completely', 'in a mixing bowl , beat cream cheese until fluffy', 'combine milk and pudding mix', 'add to cream cheese and mix well', 'fold in whipped topping and raspberries', 'reserve a large dollop of filling for garnish', 'place one cake layer on a serving plate', 'spread with half of the filling', 'do not cover sides of cake , just the top as this is made to look like a torte , not a frosted cake', 'repeat layers', 'top with remaining cake', ""dust with confectioner's sugar"", 'mound the reserved filling in the center and arrange raspberries in the middle', 'garnish with fresh mint on top if desired', 'store in refrigerator', 'this is because the strawberries will""bleed"" into the filling and become mushy', 'it will still taste great , it will just look a bit unattractive']","you won't believe how easy this cake is to make. when you present it to your guests, i promise you will receive ""oohs and aahs"". it is beautiful to look at and absolutely scrumptious to eat. my father is not quick to dole out compliments on food, he said this was the best cake he has ever had. when strawberries are in peak season i often substitute them for the raspberries.","['chocolate cake mix', 'eggs', 'oil', 'water', 'cream cheese', 'milk', 'vanilla instant pudding mix', 'frozen whipped topping', 'fresh raspberries', ""confectioners' sugar"", 'of fresh mint', 'raspberries']",12,4.945945945945946,37,4.841285746927722,433.1,40.0,120.0,23.0,12.0,53.0,15.0,Non-veg,Dinner,Other,3.25
rr s caramelized onions,209735,35,145489,2007-02-06,"['60-minutes-or-less', 'time-to-make', 'main-ingredient', 'preparation', 'low-protein', 'vegetables', 'dietary', 'low-sodium', 'low-calorie', 'low-carb', 'low-in-something', 'onions']",4,"['in a large skillet , melt the butter in the olive oil over medium-high heat', 'add the onions , the salt and pepper', 'cook , stirring constantly , until the onions begin to soften , about 5 minutes', 'stir in the sugar and cook , scraping the browned bits off the bottom of the pan frequently , until the onions are golden brown , about 20 minutes']","these are a great condiment - scatter over a pizza, chop & stir into mashed potatoes, toss with pasta & parmesan cheese!  the possibilities are endless.","['butter', 'extra virgin olive oil', 'onions', 'salt', 'pepper', 'sugar']",6,4.966666666666667,30,4.838439598940391,129.0,12.0,28.0,4.0,3.0,16.0,4.0,Veg,Other,Other,2.05
mexican coffee  caf mexicano,171163,5,242766,2006-06-02,"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'north-american', 'for-1-or-2', 'low-protein', 'healthy', 'beverages', 'mexican', 'easy', 'low-fat', 'chocolate', 'dietary', 'low-sodium', 'low-cholesterol', 'low-saturated-fat', 'low-in-something', 'number-of-servings', '3-steps-or-less']",4,"['place kahla , brandy , chocolate syrup and cinnamon in a coffee cup or mug', 'fill with hot coffee', 'stir to blend', 'top with sweetened whipped cream']","posted for the zaar world tour 2006-mexico.
this drink is so yummy and definitely warms you up on a cold day.","['kahlua', 'brandy', 'chocolate syrup', 'ground cinnamon', 'hot coffee', 'sweetened whipped cream']",6,4.944444444444445,36,4.837758763526116,156.4,0.0,61.0,1.0,0.0,1.0,5.0,Veg,Beverage,North-American,1.75
magic white sauce  and variations,92008,20,121684,2004-05-27,"['30-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'sauces', 'condiments-etc', 'eggs-dairy', 'stove-top', 'dietary', 'savory-sauces', 'equipment']",16,"['pour milk into a saucepan', 'add all other ingredients', 'place pan over a medium heat , and , using a wire balloon whisk , whisk sauce constantly until butter melts', 'be sure to work the whisk into the edges of the pan to incorporate all the flour', 'whisk frequently until the mixture comes to a boil', 'reduce heat to low and simmer for about 5 minutes , stirring occasionally , until the sauce reaches the desired consistency', 'taste and add extra seasoning if required', 'variations: mustard sauce make sauce as per recipe , but use 1 cups milk and cup chicken stock', 'along with the salt , add 2 teaspoons mustard powder , teaspoon onion powder and substitute a good pinch of cayenne pepper for the black pepper', 'stir in 1 teaspoon lemon juice when sauce is completed', 'cheese sauce make sauce as per recipe , but use 1 cup milk and 1 cup cream', 'along with the salt , add a good pinch of nutmeg and substitute a good pinch of cayenne pepper for black pepper', 'at the simmering stage stir in 75g- 100g grated tasty cheese', 'stir in 1 teaspoon lemon juice when sauce is completed', 'parsley sauce make sauce as per recipe , but use 1 cups milk and cup cream', 'when sauce is finished , mix in 4 tablespoons finely chopped parsley and 1 teaspoon lemon juice']","sick of lumpy sauce? hate making that flour and butter roux? here's the answer! this is the easiest version of white sauce ever. you won’t believe it works until you try it! you will need a wire balloon whisk for this recipe and you must make sure that all ingredients are cold (or at least at room temperature) to begin with. thanks to english food writer, delia smith, for discovering this all-in-one method. the following are my simplified adaptations for basic white sauce, along with variations for mustard sauce, cheese sauce (mornay sauce) and parsley sauce.","['milk', 'butter', 'plain flour', 'salt', 'black pepper']",5,5.0,23,4.834348261208601,627.0,76.0,0.0,44.0,23.0,155.0,11.0,Veg,Other,Other,2.65
```

### Step 2: Initialize `RagSearchEngine`
Use the `setup()` function to set up the `RagSearchEngine` with your data and configuration.

**Example code**:
```python
from pathlib import Path
from ragsearch import setup

# Define your data path and configuration parameters
data_path = Path("path/to/your/sample_data.csv")
llm_api_key = "your-cohere-api-key"
llm_model_name = "large"  # Adjust based on your model
vector_env = "production"
embedding_dim = 768  # Set this according to your embedding model's output dimension

# Initialize the RagSearchEngine
rag_engine = setup(data_path, llm_api_key, llm_model_name, vector_env, embedding_dim)
```

### Step 3: Run a Search Query
Once the `RagSearchEngine` is initialized, you can perform natural language searches.

**Example code**:
```python
query = "Find recipes with chicken"
results = rag_engine.search(query, top_k=5)

for result in results:
    print("Result:", result['metadata'])
```

## Running the Web Interface
### Step 1: Start the Flask Server
Run `web_interface.py` to start the web server:

```bash
python web_interface.py
```

### Step 2: Access the Web Interface
Open your browser and navigate to:

```
http://localhost:5000
```

### Step 3: Interact with the Web Interface
- Enter a search query in the input field.
- Click the **Submit** button.
- View the results displayed on the page.

## Testing the Package
### Running Unit Tests
Ensure your package functions as expected by running `pytest`:

```bash
poetry run pytest
```

## Advanced Usage and Customization
### Changing the Embedding Model
Modify the `llm_model_name` parameter in `setup()` to use different models, e.g., "large" or "small".

### Adding More Metadata
Include additional columns in your data for more detailed results.

### Customizing the Web Interface
Edit `index.html` in the `templates` directory to adjust the UI layout or add more user features.

## Troubleshooting
- **`AssertionError: d == self.d`**: Ensure the embedding dimension (`embedding_dim`) matches the output dimension from your embedding model.
- **`TypeError: embed() takes 1 positional argument`**: Use the correct keyword argument format for `embed()` based on your `cohere` version.

## Deployment Tips
- **Deploying to a Server**: Use services like Heroku, AWS, or Docker.

## Contributing
Feel free to contribute to this project by submitting issues, feature requests, or pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

